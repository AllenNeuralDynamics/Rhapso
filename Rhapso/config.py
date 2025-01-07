class Config:
    # AWS S3 Bucket Config
    BUCKET_NAME = 'rhapso-example-data-zarr'
    REGION = 'us-east-2'
    XMLFILENAME = 'dataset.xml'
    BASE_PATH = 'exaSPIM_708365_2024-04-29_12-46-15/SPIM.ome.zarr'
    
    # XML Paths
    VIEWSETUPSXPATH = './SequenceDescription/ViewSetups'
    TIMEPOINTSXPATH = './SequenceDescription/Timepoints'
    MISSINGVIEWSXPATH = './SequenceDescription/MissingViews'
    IMAGELOADERXPATH = './SequenceDescription/ImageLoader'
    ZGROUPSXPATH = './SequenceDescription/ImageLoader/zgroups'
    VIEWREGISTRATIONSXPATH = './ViewRegistrations'
    VIEWINTERESTPOINTSXPATH = './ViewInterestPoints'
