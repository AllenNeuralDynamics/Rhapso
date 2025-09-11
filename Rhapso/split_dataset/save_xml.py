import numpy as np
import boto3
import re
from xml.etree import ElementTree as ET

class SaveXML:
    def __init__(self, data_global, new_split_interest_points, self_definition, xml_file, xml_output_path):
        self.data_global = data_global
        self.new_split_interest_points = new_split_interest_points
        self.self_definition = self_definition
        self.xml_file = xml_file
        self.xml_output_path = xml_output_path
    
    def wrap_image_loader_for_split(self, xml: str) -> str:
        root = ET.fromstring(xml)

        def tn(el): return el.tag.split('}')[-1]
        def find_one(tag):
            el = root.find(f'.//{{*}}{tag}')
            return el if el is not None else root.find(tag)

        seq = find_one('SequenceDescription')
        if seq is None:
            return xml  

        # find the first immediate ImageLoader under SequenceDescription
        loaders = [ch for ch in list(seq) if tn(ch) == 'ImageLoader']
        if not loaders:
            return xml

        inner = loaders[0]

        fmt = (inner.get('format') or '').lower()
        if fmt == 'split.viewerimgloader':
            return xml
        
        # handle the case where the *outer* wrapper already exists
        if any(tn(ch) == 'ImageLoader' for ch in list(inner)) and fmt.startswith('bdv'):
            return xml

        # wrap the current loader
        idx = list(seq).index(inner)
        seq.remove(inner)
        wrapper = ET.Element('ImageLoader', {'format': 'split.viewerimgloader'})
        wrapper.append(inner)
        seq.insert(idx, wrapper)

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

        def tagname(el): 
            return el.tag.split('}')[-1]

        def find_one(tag):
            el = root.find(f'.//{{*}}{tag}')
            if el is None:
                el = root.find(tag)
            return el

        seq  = find_one('SequenceDescription')
        regs = find_one('ViewRegistrations')
        setup_ids = find_one('SetupIds')
        if setup_ids is None:
            setup_ids = ET.Element('SetupIds')
            kids = list(root)
            insert_idx = len(kids)
            if regs is not None and regs in kids:
                insert_idx = kids.index(regs)
            elif seq is not None and seq in kids:
                insert_idx = kids.index(seq) + 1
            root.insert(insert_idx, setup_ids)

        view_setups = None
        for ch in list(root):
            if tagname(ch) == 'ViewSetups':
                view_setups = ch
                break
        if view_setups is None:
            view_setups = find_one('ViewSetups')
        if view_setups is None:
            view_setups = ET.Element('ViewSetups')
            kids = list(root)
            after_idx = -1
            for i, ch in enumerate(kids):
                if tagname(ch) in ('ImageLoader', 'SequenceDescription'):
                    after_idx = i
            root.insert(after_idx + 1 if after_idx >= 0 else len(kids), view_setups)

        def _norm_id(raw):
            if isinstance(raw, (tuple, list)):
                if len(raw) >= 2:
                    return int(raw[1])
                return int(raw[0])
            return int(raw)

        target_ids = set(_norm_id(v['new_view']) for v in self.self_definition)

        # ViewSetup cleanup
        for child in list(view_setups):
            if tagname(child) != 'ViewSetup':
                continue
            id_el = child.find('id') or child.find('{*}id')
            if id_el is not None and id_el.text:
                try:
                    if int(id_el.text.strip()) in target_ids:
                        view_setups.remove(child)
                except Exception:
                    pass

        # SetupIdDefinition cleanup
        for sid in list(setup_ids):
            if tagname(sid) != 'SetupIdDefinition':
                continue
            nid_el = sid.find('NewId') or sid.find('{*}NewId')
            if nid_el is not None and nid_el.text:
                try:
                    if int(nid_el.text.strip()) in target_ids:
                        setup_ids.remove(sid)
                except Exception:
                    pass

        for view in self.self_definition:
            new_id = _norm_id(view['new_view'])
            old_id = _norm_id(view['old_view'])
            angle  = view['angle']
            channel = view['channel']
            illumination = view['illumination']
            tile = new_id
            voxel_unit = view['voxel_unit']
            voxel_size = view['voxel_dim'] 

            mins = np.array(view["interval"][0], dtype=np.int64)
            maxs = np.array(view["interval"][1], dtype=np.int64)
            size = (maxs - mins + 1).tolist()

            # <SetupIds>/<SetupIdDefinition>
            def_el = ET.SubElement(setup_ids, 'SetupIdDefinition')
            ET.SubElement(def_el, 'NewId').text = str(new_id)
            ET.SubElement(def_el, 'OldId').text = str(old_id)
            ET.SubElement(def_el, 'min').text   = f"{int(mins[0])} {int(mins[1])} {int(mins[2])}"
            ET.SubElement(def_el, 'max').text   = f"{int(maxs[0])} {int(maxs[1])} {int(maxs[2])}"

            # <ViewSetups>/<ViewSetup>
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

        try:
            ET.indent(root, space="  ")
        except Exception:
            pass

        return ET.tostring(root, encoding='unicode')
    
    def save_setup_id_definition_to_xml(self, xml):
        root = ET.fromstring(xml)

        # find existing nodes (namespace-agnostic)
        def tagname(el): return el.tag.split('}')[-1]
        children = list(root)
        regs_idx = next((i for i, ch in enumerate(children) if tagname(ch) == 'ViewRegistrations'), None)
        seq_idx  = next((i for i, ch in enumerate(children) if tagname(ch) == 'SequenceDescription'), None)
        setup_ids = next((ch for ch in children if tagname(ch) == 'SetupIds'), None)

        # create/position <SetupIds>
        if setup_ids is None:
            setup_ids = ET.Element('SetupIds')
            insert_idx = regs_idx if regs_idx is not None else ((seq_idx + 1) if seq_idx is not None else len(children))
            root.insert(insert_idx, setup_ids)
        else:
            setup_ids.clear()  

        for view in self.self_definition:
            new_id = view['new_view']
            old_id = view['old_view']
            min_bound = view['interval'][0]
            max_bound = view['interval'][1]
            
            nid = int(new_id[1] if isinstance(new_id, (tuple, list)) else new_id)
            oid = int(old_id[1] if isinstance(old_id, (tuple, list)) else old_id)

            def_el = ET.SubElement(setup_ids, 'SetupIdDefinition')
            ET.SubElement(def_el, 'NewId').text = str(nid)
            ET.SubElement(def_el, 'OldId').text = str(oid)
            ET.SubElement(def_el, 'min').text   = f"{int(min_bound[0])} {int(min_bound[1])} {int(min_bound[2])}"
            ET.SubElement(def_el, 'max').text   = f"{int(max_bound[0])} {int(max_bound[1])} {int(max_bound[2])}"
        
        try:
            ET.indent(root, space="  ")
        except Exception:
            pass

        return ET.tostring(root, encoding='unicode')

    def run(self):
        xml = self.save_setup_id_definition_to_xml(self.xml_file)
        xml = self.save_setup_id_to_xml(xml)
        xml = self.save_view_registrations_to_xml(xml)
        xml = self.save_view_interest_points(xml)
        xml = self.wrap_image_loader_for_split(xml)

        if self.xml_output_path:
            if self.xml_output_path.startswith("s3://"):
                bucket, key = self.xml_output_path.replace("s3://", "", 1).split("/", 1)
                boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=xml.encode('utf-8'))
            else:
                with open(self.xml_output_path, "w", encoding="utf-8") as f:
                    f.write(xml)