import numpy as np
import boto3
import re
import copy
from xml.etree import ElementTree as ET

class SaveXML:
    def __init__(self, data_global, new_split_interest_points, self_definition, xml_file, xml_output_path):
        self.data_global = data_global
        self.new_split_interest_points = new_split_interest_points
        self.self_definition = self_definition
        self.xml_file = xml_file
        self.xml_output_path = xml_output_path
    
    def save_tile_attributes_to_xml(self, xml):
        root = ET.fromstring(xml)

        def tagname(el):
            return el.tag.split('}')[-1]

        def find_one(tag):
            el = root.find(f'.//{{*}}{tag}')
            if el is None:
                el = root.find(tag)
            return el

        def _norm_id(raw):
            if isinstance(raw, (tuple, list)):
                return int(raw[1] if len(raw) > 1 else raw[0])
            return int(raw)

        # --- find ALL ViewSetups blocks ---
        view_setups_all = root.findall('.//{*}ViewSetups')
        if not view_setups_all:
            return xml  # nothing to do

        # Outer split ViewSetups = last, inner original = first non-outer
        outer_vs = view_setups_all[-1]
        inner_vs = None
        for vs in view_setups_all:
            if vs is not outer_vs:
                inner_vs = vs
                break

        # --- collect existing Attributes on OUTER ---
        children = list(outer_vs)
        attr_by_name = {}
        for ch in children:
            if tagname(ch) == 'Attributes':
                nm = ch.get('name')
                if nm:
                    attr_by_name[nm] = ch

        # --- ensure CHANNEL attributes (can still be cloned from inner) ---
        if 'channel' not in attr_by_name and inner_vs is not None:
            for ch in list(inner_vs):
                if tagname(ch) != 'Attributes':
                    continue
                nm = ch.get('name')
                if nm == 'channel':
                    cloned = copy.deepcopy(ch)
                    outer_vs.append(cloned)
                    attr_by_name['channel'] = cloned
                    break

        # --- build/overwrite ILLUMINATION attributes: old_tile_0..N ---
        illum_attrs = attr_by_name.get('illumination')
        if illum_attrs is None:
            illum_attrs = ET.Element('Attributes', {'name': 'illumination'})
            outer_vs.append(illum_attrs)
            attr_by_name['illumination'] = illum_attrs
        else:
            # clear existing <Illumination> entries
            for ch in list(illum_attrs):
                illum_attrs.remove(ch)

        # unique original tile ids from old_view
        orig_tile_ids = sorted({_norm_id(v['old_view']) for v in self.self_definition})

        for tid in orig_tile_ids:
            illum_el = ET.SubElement(illum_attrs, 'Illumination')
            ET.SubElement(illum_el, 'id').text = str(tid)
            ET.SubElement(illum_el, 'name').text = f"old_tile_{tid}"

        # --- ensure ANGLE attributes: a single Angle id=0/name=0 ---
        angle_attrs = attr_by_name.get('angle')
        if angle_attrs is None:
            # try clone from inner if it exists
            if inner_vs is not None:
                for ch in list(inner_vs):
                    if tagname(ch) == 'Attributes' and ch.get('name') == 'angle':
                        angle_attrs = copy.deepcopy(ch)
                        outer_vs.append(angle_attrs)
                        break
            # if no inner angle, synthesize default
            if angle_attrs is None:
                angle_attrs = ET.Element('Attributes', {'name': 'angle'})
                angle_el = ET.SubElement(angle_attrs, 'Angle')
                ET.SubElement(angle_el, 'id').text = "0"
                ET.SubElement(angle_el, 'name').text = "0"
                outer_vs.append(angle_attrs)
        else:
            # if it exists but has no <Angle>, make one
            has_angle = any(tagname(ch) == 'Angle' for ch in angle_attrs)
            if not has_angle:
                angle_el = ET.SubElement(angle_attrs, 'Angle')
                ET.SubElement(angle_el, 'id').text = "0"
                ET.SubElement(angle_el, 'name').text = "0"

        attr_by_name['angle'] = angle_attrs

        # ---- find or create <Attributes name="tile"> under OUTER <ViewSetups> ----
        children = list(outer_vs)
        tile_attrs = None
        insert_idx = len(children)  # default: append at end

        for i, ch in enumerate(children):
            if tagname(ch) == 'Attributes':
                name_attr = ch.get('name')
                # remember existing tile attributes if present
                if name_attr == 'tile':
                    tile_attrs = ch
                # prefer to insert tile after channel attributes if we create it
                if name_attr == 'channel':
                    insert_idx = i + 1

        if tile_attrs is None:
            tile_attrs = ET.Element('Attributes', {'name': 'tile'})
            outer_vs.insert(insert_idx, tile_attrs)

        # ---- figure out which tile ids (new_view ids) we care about ----
        target_ids = {_norm_id(v['new_view']) for v in self.self_definition}

        # Remove existing Tile entries for those ids (so we can rewrite cleanly)
        for child in list(tile_attrs):
            if tagname(child) != 'Tile':
                continue
            id_el = child.find('id') or child.find('{*}id')
            if id_el is None or not id_el.text:
                continue
            try:
                if int(id_el.text.strip()) in target_ids:
                    tile_attrs.remove(child)
            except Exception:
                pass

        # ---- build a map: setup_id -> (tx, ty, tz) from 'Image Splitting' ----
        view_regs = find_one('ViewRegistrations')
        tile_locations = {}

        if view_regs is not None:
            # iterate over ViewRegistration elements, namespace-agnostic
            for vr in view_regs.findall('.//{*}ViewRegistration'):
                setup_attr = vr.get('setup')
                if setup_attr is None:
                    continue
                try:
                    setup_id = int(setup_attr)
                except ValueError:
                    continue

                if setup_id not in target_ids:
                    continue

                # find the Image Splitting transform
                for vt in vr.findall('./{*}ViewTransform'):
                    name_el = vt.find('Name') or vt.find('{*}Name')
                    if name_el is None:
                        continue
                    if (name_el.text or '').strip().lower() != 'image splitting':
                        continue

                    aff_el = vt.find('affine') or vt.find('{*}affine')
                    if aff_el is None or not aff_el.text:
                        continue

                    nums = re.findall(
                        r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?',
                        aff_el.text
                    )

                    # 3x4 affine: we expect at least 12 numbers
                    if len(nums) >= 12:
                        tx, ty, tz = map(float, (nums[3], nums[7], nums[11]))
                    elif len(nums) >= 3:
                        # fallback: last 3 numbers
                        tx, ty, tz = map(float, nums[-3:])
                    else:
                        tx = ty = tz = 0.0

                    tile_locations[setup_id] = (tx, ty, tz)
                    break  # stop after first 'Image Splitting' for this VR

        # ---- create Tile entries for each new_view ----
        for view in self.self_definition:
            new_id = _norm_id(view['new_view'])

            if new_id in tile_locations:
                loc = tile_locations[new_id]
            else:
                # Fallback: use min bound of interval if we didn't find an image splitting transform
                mins = np.array(view['interval'][0], dtype=float)
                loc = (float(mins[0]), float(mins[1]), float(mins[2]))

            tile_el = ET.SubElement(tile_attrs, 'Tile')
            ET.SubElement(tile_el, 'id').text = str(new_id)
            ET.SubElement(tile_el, 'name').text = str(new_id)
            ET.SubElement(tile_el, 'location').text = f"{loc[0]:.1f} {loc[1]:.1f} {loc[2]:.1f}"

        # ---- reorder children in OUTER <ViewSetups>:
        # all <ViewSetup> first, then <Attributes> in illumination, channel, tile, angle order ----
        children = list(outer_vs)

        viewsetup_children = [ch for ch in children if tagname(ch) == 'ViewSetup']
        attr_children = [ch for ch in children if tagname(ch) == 'Attributes']
        other_children = [ch for ch in children if tagname(ch) not in ('ViewSetup', 'Attributes')]

        # desired attributes order
        attr_order = {'illumination': 0, 'channel': 1, 'tile': 2, 'angle': 3}

        def _attr_sort_key(el):
            name = el.get('name', '')
            return attr_order.get(name, 99)

        attr_children.sort(key=_attr_sort_key)

        # Clear existing children and re-append in desired order
        for ch in children:
            outer_vs.remove(ch)

        for ch in viewsetup_children + attr_children + other_children:
            outer_vs.append(ch)

        try:
            ET.indent(root, space="  ")
        except Exception:
            pass

        return ET.tostring(root, encoding='unicode')

    def wrap_image_loader_for_split(self, xml: str) -> str:
        """
        Wrap the top-level ImageLoader in <ImageLoader format="split.viewerimgloader">
        and move the ORIGINAL ViewSetups/Timepoints/MissingViews into an inner
        """
        root = ET.fromstring(xml)

        def tn(el):
            return el.tag.split('}')[-1]

        def find_one(tag):
            el = root.find(f'.//{{*}}{tag}')
            return el if el is not None else root.find(tag)

        seq = None
        # Prefer the top-level SequenceDescription (direct child of root)
        for ch in list(root):
            if tn(ch) == 'SequenceDescription':
                seq = ch
                break
        if seq is None:
            seq = find_one('SequenceDescription')
        if seq is None:
            return xml

        children = list(seq)

        # Find the first immediate ImageLoader under SequenceDescription
        base_loader = None
        base_loader_idx = None
        for i, ch in enumerate(children):
            if tn(ch) == 'ImageLoader':
                base_loader = ch
                base_loader_idx = i
                break

        if base_loader is None:
            return xml

        fmt = (base_loader.get('format') or '').lower()
        # Already wrapped; assume layout is correct and do nothing
        if fmt == 'split.viewerimgloader':
            return xml

        # Collect ORIGINAL ViewSetups / Timepoints / MissingViews that are siblings
        orig_viewsetups = None
        orig_timepoints = None
        orig_missingviews = None

        for ch in children[base_loader_idx + 1:]:
            name = tn(ch)
            if name == 'ViewSetups':
                orig_viewsetups = ch
            elif name == 'Timepoints':
                orig_timepoints = ch
            elif name == 'MissingViews':
                orig_missingviews = ch

        # Remove them from the outer SequenceDescription
        for node in (orig_viewsetups, orig_timepoints, orig_missingviews):
            if node is not None and node in seq:
                seq.remove(node)

        # Remove the original loader from seq
        seq.remove(base_loader)

        # Build wrapper <ImageLoader format="split.viewerimgloader">
        wrapper = ET.Element('ImageLoader', {'format': 'split.viewerimgloader'})
        # First child: original loader
        wrapper.append(base_loader)

        # Inner <SequenceDescription> that holds the original ViewSetups/Timepoints/MissingViews
        inner_seq = ET.Element('SequenceDescription')
        if orig_viewsetups is not None:
            inner_seq.append(orig_viewsetups)
        if orig_timepoints is not None:
            inner_seq.append(orig_timepoints)
        if orig_missingviews is not None:
            inner_seq.append(orig_missingviews)

        wrapper.append(inner_seq)

        # Insert wrapper where the original loader was
        seq.insert(base_loader_idx, wrapper)

        try:
            ET.indent(root, space="  ")
        except Exception:
            pass

        return ET.tostring(root, encoding='unicode')
    
    def save_view_interest_points(self, xml):
        root = ET.fromstring(xml)

        def find_one(tag):
            el = root.find(f'.//{{*}}{tag}')
            if el is None:
                el = root.find(tag)
            return el

        def parse_tp_setup(n5_path, key=None):
            if isinstance(n5_path, str):
                m = re.search(r'tpId_(\d+)_viewSetupId_(\d+)', n5_path)
                if m:
                    return str(m.group(1)), int(m.group(2))
            if isinstance(key, (tuple, list)) and len(key) == 2:
                t, s = key
                return str(t), int(s)
            if isinstance(key, str):
                m = re.search(r'timepoint:\s*(\d+).*setup:\s*(\d+)', key)
                if m:
                    return str(m.group(1)), int(m.group(2))
            return "0", 0

        # Ensure <ViewInterestPoints> exists
        vip = find_one('ViewInterestPoints')
        if vip is None:
            vip = ET.Element('ViewInterestPoints')
            root.append(vip)

        # Remove ALL existing entries
        for child in list(vip):
            vip.remove(child)

        # Write new entries
        seen = set()
        for key, label_entries in self.new_split_interest_points.items():
            for entry in label_entries:
                if isinstance(entry, dict) and 'ip_list' in entry:
                    label = entry.get('label') or entry.get('key') or entry.get('name')
                    ip_list = entry['ip_list']
                elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                    label, ip_list = entry
                else:
                    ip_list = entry
                    label = None

                # Pull fields
                n5_path = ip_list.get('xml_n5_path') or ip_list.get('path') or ''
                params = ip_list.get('parameters', None)
                if label is None and isinstance(n5_path, str) and '/' in n5_path:
                    label = n5_path.rsplit('/', 1)[-1] 

                t, s = parse_tp_setup(n5_path, key)
                label = "" if label is None else str(label)

                sig = (t, s, label, n5_path, params)
                if sig in seen:
                    continue
                seen.add(sig)

                attrs = {
                    'timepoint': str(t),
                    'setup': str(s),
                    'label': label,
                }
                if params is not None:
                    attrs['params'] = str(params)

                elem = ET.SubElement(vip, 'ViewInterestPointsFile', attrs)
                elem.text = n5_path

        try:
            ET.indent(root, space="  ")
        except Exception:
            pass
        
        return ET.tostring(root, encoding='unicode')

    def save_view_registrations_to_xml(self, xml):
        root = ET.fromstring(xml)

        def tagname(el): 
            return el.tag.split('}')[-1]

        def find_one(tag):
            el = root.find(f'.//{{*}}{tag}')
            if el is None:
                el = root.find(tag)
            return el

        # Find or create <ViewRegistrations>
        view_regs = find_one('ViewRegistrations')
        if view_regs is None:
            view_regs = ET.Element('ViewRegistrations')
            root.append(view_regs)

        # --- only OLD ids here ---
        targets = set()
        for view in self.self_definition:
            if 'old_view' in view:
                tp_str, setup_val = view['old_view']
                t = str(tp_str)
                s = int(setup_val)
            else:
                t = str(view.get('timepoint', '0'))
                s = int(view['setup'])
            targets.add((t, s))

        # Remove existing ViewRegistration nodes for those pairs
        for vr in list(view_regs):
            if tagname(vr) != 'ViewRegistration':
                continue
            tp = vr.get('timepoint')
            st = vr.get('setup')
            if tp is not None and st is not None and (tp, int(st)) in targets:
                view_regs.remove(vr)

        # Rebuild registrations (only OLD ids)
        for view in self.self_definition:
            tp_str, setup_val = view['old_view']
            t = str(tp_str)
            setup_id = str(view['new_view'][1])
            old_models = list(view.get('old_models', []))

            vr = ET.SubElement(view_regs, 'ViewRegistration', {
                'timepoint': t,
                'setup': setup_id,
            })

            for tr in old_models:
                vt = ET.SubElement(vr, 'ViewTransform', {'type': tr.get('type', 'affine')})
                nm = ET.SubElement(vt, 'Name')
                nm.text = str(tr.get('name', ''))

                aff = ET.SubElement(vt, 'affine')
                raw = tr.get('affine', '')
                txt = raw.get('string', raw) if isinstance(raw, dict) else raw
                nums = re.findall(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?', str(txt))
                aff.text = ' '.join(nums[:12] if len(nums) >= 12 else nums)

                if (nm.text or '').strip().lower() == 'image splitting':
                    aff.text = ' '.join(f'{float(x):.1f}' for x in nums[:12])

        try:
            ET.indent(root, space="  ")
        except Exception:
            pass

        return ET.tostring(root, encoding='unicode')

    def save_setup_id_to_xml(self, xml):
        root = ET.fromstring(xml)

        def tn(el):
            return el.tag.split('}')[-1]

        # Find top-level SequenceDescription
        outer_seq = None
        for ch in list(root):
            if tn(ch) == 'SequenceDescription':
                outer_seq = ch
                break
        if outer_seq is None:
            outer_seq = root.find('.//{*}SequenceDescription')
        if outer_seq is None:
            return xml

        # Find or create OUTER <ViewSetups> under SequenceDescription
        view_setups = None
        for ch in list(outer_seq):
            if tn(ch) == 'ViewSetups':
                view_setups = ch
                break

        if view_setups is None:
            view_setups = ET.Element('ViewSetups')
            children = list(outer_seq)
            insert_idx = len(children)
            for i, ch in enumerate(children):
                if tn(ch) == 'ImageLoader':
                    insert_idx = i + 1
            outer_seq.insert(insert_idx, view_setups)

        # Helper to normalize ids
        def _norm_id(raw):
            if isinstance(raw, (tuple, list)):
                if len(raw) >= 2:
                    return int(raw[1])
                return int(raw[0])
            return int(raw)

        target_ids = {_norm_id(v['new_view']) for v in self.self_definition}

        # Remove any existing ViewSetup with those ids (outer only)
        for child in list(view_setups):
            if tn(child) != 'ViewSetup':
                continue
            id_el = child.find('id') or child.find('{*}id')
            if id_el is not None and id_el.text:
                try:
                    if int(id_el.text.strip()) in target_ids:
                        view_setups.remove(child)
                except Exception:
                    pass

        # (Re)build ViewSetups for each new split view
        for view in self.self_definition:
            new_id = _norm_id(view['new_view'])
            # old_id = _norm_id(view['old_view'])   # not strictly needed here

            angle       = view['angle']
            channel     = view['channel']
            illumination = view['illumination']
            tile        = new_id
            voxel_unit  = view['voxel_unit']
            voxel_size  = view['voxel_dim']

            mins = np.array(view["interval"][0], dtype=np.int64)
            maxs = np.array(view["interval"][1], dtype=np.int64)
            size = (maxs - mins + 1).tolist()

            vs = ET.SubElement(view_setups, 'ViewSetup')
            ET.SubElement(vs, 'id').text   = str(new_id)
            ET.SubElement(vs, 'size').text = f"{int(size[0])} {int(size[1])} {int(size[2])}"

            vx = ET.SubElement(vs, 'voxelSize')
            ET.SubElement(vx, 'unit').text = str(voxel_unit)
            if isinstance(voxel_size, str):
                ET.SubElement(vx, 'size').text = voxel_size.strip()
            else:
                ET.SubElement(vx, 'size').text = " ".join(str(x) for x in voxel_size)

            attrs = ET.SubElement(vs, 'attributes')
            ET.SubElement(attrs, 'illumination').text = str(int(illumination))
            ET.SubElement(attrs, 'channel').text      = str(int(channel))
            ET.SubElement(attrs, 'tile').text         = str(int(tile))
            ET.SubElement(attrs, 'angle').text        = str(int(angle))

        # Ensure outer <Timepoints> exists
        outer_timepoints = None
        for ch in list(outer_seq):
            if tn(ch) == 'Timepoints':
                outer_timepoints = ch
                break
        if outer_timepoints is None:
            outer_timepoints = ET.Element('Timepoints', {'type': 'pattern'})
            ip = ET.SubElement(outer_timepoints, 'integerpattern')
            ip.text = "0"
            # place right after ViewSetups
            children = list(outer_seq)
            insert_idx = children.index(view_setups) + 1 if view_setups in children else len(children)
            outer_seq.insert(insert_idx, outer_timepoints)

        # Ensure outer <MissingViews> exists
        outer_missing = None
        for ch in list(outer_seq):
            if tn(ch) == 'MissingViews':
                outer_missing = ch
                break
        if outer_missing is None:
            outer_missing = ET.Element('MissingViews')
            outer_seq.append(outer_missing)

        try:
            ET.indent(root, space="  ")
        except Exception:
            pass

        return ET.tostring(root, encoding='unicode')
    
    def save_setup_id_definition_to_xml(self, xml):
        """
        Create/overwrite <SetupIds> for the split views.

        In the desired final layout, SetupIds lives inside:
          <SequenceDescription>
            <ImageLoader format="split.viewerimgloader">
              ...
              <SequenceDescription> ... </SequenceDescription>
              <SetupIds> ... </SetupIds>   <-- here
            </ImageLoader>
            ...
          </SequenceDescription>
        """
        root = ET.fromstring(xml)

        def tn(el):
            return el.tag.split('}')[-1]

        # Find top-level SequenceDescription
        outer_seq = None
        for ch in list(root):
            if tn(ch) == 'SequenceDescription':
                outer_seq = ch
                break
        if outer_seq is None:
            outer_seq = root.find('.//{*}SequenceDescription')
        if outer_seq is None:
            return xml

        # Find the wrapper ImageLoader format="split.viewerimgloader"
        wrapper = None
        for ch in list(outer_seq):
            if tn(ch) == 'ImageLoader' and (ch.get('format') or '').lower() == 'split.viewerimgloader':
                wrapper = ch
                break

        # If wrapper not found, fall back to old behavior (root-level SetupIds)
        parent_for_setupids = wrapper if wrapper is not None else root
        children = list(parent_for_setupids)

        # Locate existing <SetupIds> under the chosen parent
        setup_ids = None
        for ch in children:
            if tn(ch) == 'SetupIds':
                setup_ids = ch
                break

        if setup_ids is None:
            setup_ids = ET.Element('SetupIds')
            if wrapper is not None:
                # Under wrapper: insert after inner SequenceDescription if present
                inner_children = list(wrapper)
                inner_seq = None
                for ich in inner_children:
                    if tn(ich) == 'SequenceDescription':
                        inner_seq = ich
                        break
                insert_idx = inner_children.index(inner_seq) + 1 if inner_seq is not None else len(inner_children)
                wrapper.insert(insert_idx, setup_ids)
            else:
                # Root-level fallback: put before <ViewRegistrations> if present
                root_children = list(root)
                regs_idx = next((i for i, ch in enumerate(root_children) if tn(ch) == 'ViewRegistrations'), None)
                insert_idx = regs_idx if regs_idx is not None else len(root_children)
                root.insert(insert_idx, setup_ids)
        else:
            # Clear existing definitions so we can rewrite
            setup_ids.clear()

        # Now populate SetupIdDefinition from self.self_definition
        for view in self.self_definition:
            new_id = view['new_view']
            old_id = view['old_view']
            min_bound = view['interval'][0]
            max_bound = view['interval'][1]

            # Normalize IDs (can be int or (tp, setup))
            nid = int(new_id[1] if isinstance(new_id, (tuple, list)) else new_id)
            oid = int(old_id[1] if isinstance(old_id, (tuple, list)) else old_id)

            def_el = ET.SubElement(setup_ids, 'SetupIdDefinition')
            ET.SubElement(def_el, 'NewId').text = str(nid)
            ET.SubElement(def_el, 'OldId').text = str(oid)
            ET.SubElement(def_el, 'min').text = f"{int(min_bound[0])} {int(min_bound[1])} {int(min_bound[2])}"
            ET.SubElement(def_el, 'max').text = f"{int(max_bound[0])} {int(max_bound[1])} {int(max_bound[2])}"

        try:
            ET.indent(root, space="  ")
        except Exception:
            pass

        return ET.tostring(root, encoding='unicode')

    def run(self):
        xml = self.xml_file
        xml = self.wrap_image_loader_for_split(xml)
        xml = self.save_setup_id_definition_to_xml(xml)
        xml = self.save_setup_id_to_xml(xml)
        xml = self.save_view_registrations_to_xml(xml)
        xml = self.save_tile_attributes_to_xml(xml)
        xml = self.save_view_interest_points(xml)

        if self.xml_output_path:
            if self.xml_output_path.startswith("s3://"):
                bucket, key = self.xml_output_path.replace("s3://", "", 1).split("/", 1)
                boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=xml.encode('utf-8'))
            else:
                with open(self.xml_output_path, "w", encoding="utf-8") as f:
                    f.write(xml)