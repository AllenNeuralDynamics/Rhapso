from scipy.spatial import cKDTree
from copy import deepcopy
import time
import random
import numpy as np
import zarr
import s3fs
import math

class SplitImages:
    def __init__(self, target_image_size, target_overlap, min_step_size, data_gloabl, n5_path, point_density, min_points, max_points, 
                 error, excludeRadius):
        self.target_image_size = target_image_size
        self.target_overlap = target_overlap
        self.min_step_size = min_step_size
        self.data_global = data_gloabl
        self.image_loader_df = data_gloabl['image_loader']
        self.view_setups_df = data_gloabl['view_setups']
        self.view_registrations_df = data_gloabl['view_registrations']
        self.view_interest_points_df = data_gloabl['view_interest_points']
        self.n5_path = n5_path
        self.point_density = point_density
        self.min_points = min_points
        self.max_points = max_points
        self.error = error
        self.exclude_radius = excludeRadius
        self.setup_definition = []

    def intersect(self, interval, other_interval):
        n = len(interval[0])
        mins = [max(interval[0][d],  other_interval[0][d]) for d in range(n)]
        maxs = [min(interval[1][d],  other_interval[1][d]) for d in range(n)]

        return (mins, maxs)
    
    def create_models(self, transform_list):
        M = np.eye(4)
        for tr in transform_list:
            A = np.fromstring(tr["affine"].replace(",", " "), sep=" ").reshape(3, 4)
            T = np.vstack([A, [0, 0, 0, 1]])
            M = M @ T  
        
        vals = M[:3, :].ravel() 
        m00,m01,m02,m03, m10,m11,m12,m13, m20,m21,m22,m23 = map(float, vals)

        model = {
            "type": "AffineTransform3D",
            "string": "3d-affine: (" + ", ".join(format(v, ".16g") for v in vals) + ")",
            "a": {
                "type": "AffineTransform3D$AffineMatrix3D",
                "m00": m00, "m01": m01, "m02": m02, "m03": m03,
                "m10": m10, "m11": m11, "m12": m12, "m13": m13,
                "m20": m20, "m21": m21, "m22": m22, "m23": m23,
                "m": [[m00, m01, m02, m03],
                    [m10, m11, m12, m13],
                    [m20, m21, m22, m23]],
            },
            "d0": {"type": "RealPoint", "string": f"({format(m00,'.16g')},{format(m10,'.16g')},{format(m20,'.16g')})",
                "n": 3, "position": [m00, m10, m20]},
            "d1": {"type": "RealPoint", "string": f"({format(m01,'.16g')},{format(m11,'.16g')},{format(m21,'.16g')})",
                "n": 3, "position": [m01, m11, m21]},
            "d2": {"type": "RealPoint", "string": f"({format(m02,'.16g')},{format(m12,'.16g')},{format(m22,'.16g')})",
                "n": 3, "position": [m02, m12, m22]},
            "ds": [[m00, m10, m20], [m01, m11, m21], [m02, m12, m22]],
        }

        return model

    def localizing_zero_min_interval_iterator(self, dimensions):
        dims = [int(d) for d in dimensions]
        n = len(dims)
        mn = [0] * n
        mx = [d - 1 for d in dims]
        steps = [1] * n
        for d in range(1, n):
            steps[d] = steps[d - 1] * dims[d - 1]
        last_index = (steps[-1] * dims[-1] - 1) if n else -1
        pos = mn.copy()
        if n:
            pos[0] = mn[0] - 1  

        return {
            "dimensions": dims,
            "index": -1,
            "last_index": last_index,
            "max": mx,
            "min": mn,
            "n": n,
            "position": pos,
            "steps": steps,  
        }

    def split_dims(self, input, i, final_size, overlap):
        dim_intervals = []
        input_min = [0, 0, 0]
        to_val = 0
        from_val = input_min[i]

        while to_val < input[i]:
            to_val = min(input[i], from_val + final_size - 1)
            dim_intervals.append((from_val, to_val))
            from_val = to_val - overlap + 1
        
        return dim_intervals
    
    def last_image_size(self, l, s, o):
        num = l - 2 * (s - o) - o
        den = s - o
        rem = num % den if num >= 0 else -((-num) % den)  
        size = o + rem
        if size < 0:
            size = l + size
        return size
    
    def distribute_intervals_fixed_overlap(self, input):
        input = list(map(int, input.split()))
        
        for i in range(len(input)):
            if self.target_image_size[i] % self.min_step_size[i] != 0:
                raise RuntimeError(f"target size {self.target_image_size[i]} not divisible by min step size {self.min_step_size[i]} for dim {i}")
            elif self.target_overlap[i] % self.min_step_size[i] != 0:
                raise RuntimeError(f"overlap {self.target_overlap[i]} not divisible by min step size {self.min_step_size[i]} for dim {i}")
            
        interval_basis = []
        for i in range(len(input)):
            dim_intervals = []
            length = input[i]

            if length <= self.target_image_size[i]:
                pass
            
            else:
                l = length
                s = self.target_image_size[i]
                o = self.target_overlap[i]
                last_image_size = self.last_image_size(l, s, o)
                
                final_size = 0
                if last_image_size != s:
                    last_size = s
                    delta = 0
                    current_last_image_size = 0

                    if last_image_size <= s // 2:
                        while True:
                            last_size += self.min_step_size[i]
                            current_last_image_size = self.last_image_size(l, last_size, o)
                            delta = last_image_size - current_last_image_size
                            last_image_size = current_last_image_size
                            if delta <= 0: break
                        
                        final_size = last_size
                    
                    else:
                        while True:
                            last_size -= self.min_step_size[i]
                            current_last_image_size = self.last_image_size(l, last_size, o)
                            delta = last_image_size - current_last_image_size
                            last_image_size = current_last_image_size
                            if delta >= 0: break
                        
                        final_size = last_size + self.min_step_size[i]
                
                else:
                    final_size = s 
                
                split_dims = self.split_dims(input, i, final_size, self.target_overlap[i])
                dim_intervals.extend(split_dims)
            
            interval_basis.append(dim_intervals)

        num_intervals = []
        for i in range(len(input)):
            num_intervals.append(len(interval_basis[i]))
        
        cursor = self.localizing_zero_min_interval_iterator(num_intervals)
        interval_list = []
        current_interval = [0, 0, 0]
        
        while cursor['index'] < cursor['last_index']:
            
            # fwd
            cursor['index'] = cursor['index'] + 1
            for i in range(cursor['n']):
                cursor['position'][i] = cursor['position'][i] + 1
                if cursor['position'][i] > cursor['max'][i]:
                    cursor['position'][i] = 0
                else:
                    break
            
            # localize
            for i in range(cursor['n']):
                current_interval[i] = cursor['position'][i]
            
            min_val = [0, 0, 0]
            max_val = [0, 0, 0]

            for i in range(len(input)):
                min_max = interval_basis[i][current_interval[i]]
                min_val[i] = min_max[0]
                max_val[i] = min_max[1]
            
            interval_list.append((min_val, max_val))
        
        return interval_list  
    
    def max_interval_spread(self, old_setups_df):
        max_val = 1
        for _, row in old_setups_df.iterrows():
            input = row['size']
            intervals = self.distribute_intervals_fixed_overlap(input)
            max_val = max(len(intervals), max_val)
        
        return max_val
    
    def is_empty(self, interval):
        if interval is None:
            return True
        mins, maxs = interval
        return any(mn > mx for mn, mx in zip(mins, maxs))
    
    def contains(self, ip, interval):
        for i in range(len(ip)):
            if ip[i] < interval[0][i] or ip[i] > interval[1][i]:
                return False
        
        return True

    def split_images(self, timepoints, interest_points, fake_label):
        old_setups_df = deepcopy(self.view_setups_df)
        old_registrations_df = deepcopy(self.view_registrations_df)

        new_to_old_setup_id = {}
        new_setup_id_to_interval = {}
        new_setups = []
        new_registrations = {}
        new_interest_points = {}

        new_id = 0
        max_interval_spread = self.max_interval_spread(old_setups_df)
        rnd = random.Random(23424459)

        for _, row in old_setups_df.iterrows():
            old_id = row['id']
            angle = row['angle']
            channel = row['channel']
            vox_dim = row['voxel_size']
            vox_unit = row['voxel_unit']
            illumination = row['illumination']
            input = row['size']
            local_new_tile_id = 0

            intervals = self.distribute_intervals_fixed_overlap(input)

            interval_to_view_setup = {}
            for i in range(len(intervals)): 
                interval = intervals[i]
                new_to_old_setup_id[new_id] = old_id
                new_setup_id_to_interval[new_id] = interval

                size = [0, 0, 0]

                for j in range(3):
                    size[j] = interval[1][j] - interval[0][j] + 1
                
                new_dim = deepcopy(size)

                location = [0, 0, 0]
                for j in range(len(interval[0])):
                    location[j] += interval[0][j]
                
                new_tile_id = int(old_id) * max_interval_spread + local_new_tile_id
                local_new_tile_id += 1
                
                new_tile = {
                    'id': new_tile_id,
                    'location': location,
                    'name': str(new_tile_id)
                }
                
                new_illum = {
                    'id': old_id,
                    'name': "old_tile_" + old_id
                }
                
                new_setup = {
                    'angle':str(angle), 
                    'attributes': {
                        'illumination': new_illum,
                        'channel': channel,
                        'tile': new_tile,
                        'angle': angle
                    },
                    'channel': str(channel),
                    'id': new_tile_id,
                    'illumination': new_illum,
                    'name': None,
                    'size': new_dim,
                    'tile': new_tile,
                    'voxelSize': {
                        'dimensions': vox_dim,
                        'unit': vox_unit
                    }
                }

                new_setups.append(new_setup)
                interval_key = (tuple(interval[0]), tuple(interval[1]))
                interval_to_view_setup[interval_key] = new_setup

                for t in timepoints:
                    old_view_id = f"timepoint: {t}, setup: {old_id}" 
                    old_vr = (old_registrations_df['timepoint'] == str(t)) & (old_registrations_df['setup'] == str(old_id))
                    transform_list = old_registrations_df.loc[old_vr, ['name', 'type', 'affine']].to_dict('records')
                    
                    mn, _ = interval 
                    translation = f"1, 0, 0, {mn[0]}, 0, 1, 0, {mn[1]}, 0, 0, 1, {mn[2]}"
                    
                    transform = {
                        'name': 'Image Splitting',
                        'affine': translation  
                    }
                    transform_list.append(transform)

                    new_view_id = {
                        'setup': new_id,
                        'timepoint': t
                    }

                    new_view_id_key = f"timepoint: {t}, setup: {new_view_id['setup']}"

                    model = self.create_models(transform_list)

                    new_view_registration = {
                        'model': model,
                        'setup': new_view_id['setup'],
                        'timepoint': t,
                        'transformList': transform_list
                    }

                    new_registrations[(new_view_id_key)] = new_view_registration

                    new_v_ip_l = []

                    old_v_ip_l = {
                        'points': interest_points[old_view_id],
                        'setup': old_id,
                        'timepoint': t,
                    }

                    id = 0
                    new_ip1 = []
                    old_ip_l1 = old_v_ip_l['points']
                    old_ip_1 = deepcopy(old_ip_l1['points'])
                    
                    for ip in old_ip_1:   
                        if self.contains(ip, interval):
                            l = deepcopy(ip)
                            for j in range(len(interval[0])):
                                l[j] -= interval[0][j]
                            
                            new_ip1.append((id, l))
                            id += 1
                    
                    new_ip_l1 = {
                        'base_directory': old_ip_l1['base_path'],
                        'corresponding_interest_points': None,
                        'interest_points': new_ip1,
                        'modified_corresponding_interest_points': None,
                        'modified_interest_points': None,
                        'n5_path': f"interestpoints.n5/tpId_{t}_viewSetupId_{new_view_id['setup']}/beads_split",
                        'xml_n5_path': f"tpId_{t}_viewSetupId_{new_view_id['setup']}/{fake_label}",
                        "parameters": old_ip_l1['parameters_split']
                    }

                    new_v_ip_l.append({
                        'label': "beads_split",
                        'ip_list': new_ip_l1
                    })
                    
                    new_ip = []
                    id = 0

                    for j in range(i):
                        other_interval = intervals[j]
                        intersection = self.intersect(interval, other_interval)
                        
                        if not self.is_empty(intersection):
                            other_setup = interval_to_view_setup[(tuple(other_interval[0]), tuple(other_interval[1]))]
                            other_view_id = f"timepoint: {t}, setup: {other_setup['id']}"
                            other_ip_list = new_interest_points[other_view_id]
                            
                            n = len(interval[0])
                            num_pixels = 1

                            for k in range(n):
                                num_pixels *= (intersection[1][k] - intersection[0][k] + 1)
                            
                            num_points = min(self.max_points, max(self.min_points,  math.ceil(self.point_density * num_pixels / (100.0*100.0*100.0))))
                            other_points = (next((x for x in other_ip_list if x.get("label") == fake_label), {"ip_list": {}})["ip_list"].get("interest_points") or [])
                            other_id = len(other_points)

                            tree2 = None
                            search2 = None

                            if self.exclude_radius > 0:
                                other_ip_global = []
                                
                                for k, ip in enumerate(other_points):
                                    l = deepcopy(ip[1])
                                    
                                    for m in range(n):
                                        l[m] = l[m] + other_interval[0][m]
                                    
                                    other_ip_global.append((k, l))

                                if len(other_ip_global) > 0:
                                    coords = np.vstack([l for _, l in other_ip_global])  
                                    tree2 = cKDTree(coords)

                                    def search2(q_point_global, radius=self.exclude_radius):
                                        idxs = tree2.query_ball_point(np.asarray(q_point_global, float), radius)
                                        return [other_ip_global[k] for k in idxs]
                                else:
                                    tree2 = None
                                    search2 = None
                            
                            else:
                                tree2 = None
                                search2 = None

                            tmp = [0.0] * n

                            for k in range(num_points):
                                p = [0.0] * n
                                op = [0.0] * n
                                
                                for d in range(n):
                                    l = rnd.random() * (intersection[1][d] - intersection[0][d] + 1) + intersection[0][d]
                                    p[d]  = (l + (rnd.random() - 0.5) * self.error) - interval[0][d]
                                    op[d] = (l + (rnd.random() - 0.5) * self.error) - other_interval[0][d]
                                    tmp[d] = l
                                
                                num_neighbors = 0
                                if self.exclude_radius > 0:
                                    tmp_ip = (0, np.asarray(tmp, dtype=float))  
                                    
                                    if search2 is not None:
                                        neighbors = search2(tmp_ip[1], self.exclude_radius) 
                                        num_neighbors += len(neighbors)
                                
                                if num_neighbors == 0:
                                    new_ip.append((id, p))
                                    other_points.append((other_id, op))
                                    id += 1
                                    other_id += 1
                            
                            next(x for x in other_ip_list if x.get("label") == fake_label)["ip_list"]["interest_points"] = other_points
                    
                    new_ip_l = {
                        'base_directory': old_ip_l1['base_path'],
                        'corresponding_interest_points': None,
                        'interest_points': new_ip,
                        'modified_corresponding_interest_points': None,
                        'modified_interest_points': None,
                        'n5_path': f"interestpoints.n5/tpId_{t}_viewSetupId_{new_view_id['setup']}/{fake_label}",
                        'xml_n5_path': f"tpId_{t}_viewSetupId_{new_view_id['setup']}/{fake_label}",
                        "parameters": old_ip_l1['parameters_fake']
                    }

                    new_v_ip_l.append({
                        'label': fake_label,
                        'ip_list': new_ip_l
                    })
            
                self.setup_definition.append({
                    'interval': interval,
                    'old_view': (t, old_id),
                    'new_view': (t, new_id),
                    'voxel_dim': vox_dim,
                    'voxel_unit': vox_unit,
                    'angle': angle,
                    'channel': channel,
                    'illumination': illumination,
                    'old_models': transform_list
                })

                new_interest_points[new_view_id_key] = new_v_ip_l
                new_id += 1

        return new_interest_points
    
    def load_interest_points(self, fake_label):
        full_path = self.n5_path + "interestpoints.n5"
        interest_points = {}

        if full_path.startswith("s3://"):
            path = full_path.rstrip("/")
            s3 = s3fs.S3FileSystem(anon=False)
            store = s3fs.S3Map(root=path, s3=s3, check=False)
            root = zarr.open(store, mode="r")            
        
        else:
            store = zarr.N5Store(full_path)
            root = zarr.open(store, mode="r") 

        for _, row in self.view_interest_points_df.iterrows():
            view_id = f"timepoint: {row['timepoint']}, setup: {row['setup']}"  
            interestpoints_prefix = f"{row['path']}/interestpoints/loc/"
            fake_path = f"tpId_{row['timepoint']}_viewSetupId_{row['setup']}/{fake_label}"
            split_path =  f"tpId_{row['timepoint']}_viewSetupId_{row['setup']}/beads_split"
            overlap_px = f"[{self.target_overlap[0]}, {self.target_overlap[1]}, {self.target_overlap[2]}]"
            
            group = root[interestpoints_prefix]
            data = group[:]
            
            interest_points[view_id] = {
                'points': data,
                'n5_path_old': row['path'],
                'base_path': full_path,
                'n5_path_split_points': split_path,
                'n5_path_fake_points': fake_path,
                'parameters_split': row['params'],
                'parameters_fake': f"Fake points for image splitting: overlapPx={overlap_px}, targetSize={self.target_image_size}, minStepSize={self.min_step_size}, optimize=true, pointDensity={self.point_density}, minPoints={self.min_points}, maxPoints={self.max_points}, error={self.error}, excludeRadius={self.exclude_radius}"
            }

        return interest_points

    def run(self):
        timepoints = set()
        for _, row in self.image_loader_df.iterrows():
            timepoints.add(row['timepoint'])
        
        fake_label = f"splitPoints_{int(time.time() * 1000)}"
        interest_points = self.load_interest_points(fake_label)
        new_split_interest_points = self.split_images(timepoints, interest_points, fake_label)

        return new_split_interest_points, self.setup_definition