# 10.28.25 multiscale edits / changes v2.0

change1: multiscale addition 
'do this first, then comment out zero resolution thing and point to this'

--- ! new ! ---
1. remove dask from 
```
        for sl in BlockedArrayWriter.gen_slices(in_array.shape, block_shape):
            block = in_array[sl]
            da.store(
                block,
                out_array,
                regions=sl,
                lock=False,
                compute=True,
                return_stored=False,
            )
```
- confirm still works without dask
2. add custom fetch/write inside just like fusion (will have to pass in dataset, two vars, instead of output_volume)
- tricky: 
at this point` for level in range(0, n_lvls):` will have to pass in dask group info per level 
- can get from 'write_ome_ngff_metadata', can pull out zarr group into and pass into ray distirbute dpart 

3. iteralte for loop approach, look at output in s3
    - dont remove, comment out when done 
4. ray locally distributed 'for sl in BlockedArrayWriter.gen_slices(in_array.shape, block_shape):' 
5.  aws distribute 

=================
change2: initial multiscale dataset improvement:

write_ome_ngff_metadata 
- writing zarr groups for each level
- later: pull out zero rez metadata

- pyramid_group = new_channel_group.create_dataset(
- skip this step, unecesary duplication, copying full Resolution
- always write to the original folder

- comment out 'pyramid_group' and uses
- add pre-fetch step: 
- instead of pyramid_group creation, assign pyramid_group to preexisitng full res zarr data in s3
- make pyramid_group an object that has access to the s3 data, pointer to group, dask array(?)
- zarr root/group point to existing s3 localization 
- in fusion: we save to existing s3 location
- copy implementation of save to s3 in fusion for multiscale

- we added prestep fetch of data in fusion, reference this 
- in fusion: instead of passing in output volume, we initialize object we can pass 

```python
store = output_volume.store
    write_root = getattr(store, "root", None) or getattr(store, "path", None)
    write_ds = output_volume.path
```
in fusion: passing in output volume,
pre existing zarr data in s3, we assign pointer to it in multiscale so we can write to it

-- next steps: reuse objects, swap out this function create_dataset
```
    # # Writing first multiscale by default
    # pyramid_group = new_channel_group.create_dataset(
    #     name="0",
    #     shape=dataset_shape,
    #     chunks=chunk_size,
    #     dtype=array.dtype,
    #     compressor=compressor,
    #     dimension_separator="/",
    #     overwrite=True,
    # )
    # fetch from s3 pointer to full rez data (full res zarr data)
```
=================

change3:
- divide installs in setup.py && update readme 
- remove unnecesary code 

temp roadmap:
- distribute multiscale 
- fix exaSpim / validate 
- add cache 