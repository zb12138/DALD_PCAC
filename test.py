
EXPNAME = 'Exp/test/'
if __name__ == '__main__':
    from EncTop import TEncTop, os
    from DecTop import TDecTop, TComPointCloud
    from testTool.testTMC import run_cmd, PCERROR_PATH
    from networkTool import IS_LIDAR
    
    if IS_LIDAR:
        inputFile = 'Data/MPEG/MPEGCat3Frame/Ford_02_q1mm/Ford_02_vox1mm-0100.ply'
    else:
        inputFile = 'Data/8iVFBv2/longdress/Ply/longdress_vox10_1051.ply'

    encoder = TEncTop()
    decoder = TDecTop()

    geoOnlyFile = EXPNAME + 'geo_only/' + os.path.basename(inputFile)
    outputFile = EXPNAME + 'decode.ply'

    print('deep entropy model:', encoder.m_deepEntroyModel.model_path, '\n'+'_'*100)

    # encoding
    print('encoding...')
    print(encoder.encode(inputFile, out_bin_dir=EXPNAME), '\n'+'_'*100)

    # create geo file
    GeoOnly = TComPointCloud()
    GeoOnly.readFromFile(inputFile, atriType="xyz")
    GeoOnly.saveToFile(geoOnlyFile)

    # decodeing
    print('decoding...')
    print(decoder.decode(inputFileName=geoOnlyFile, bin_dir=EXPNAME, decodeFileName=outputFile), '\n'+'_'*100)

    print('pc_error...')
    # pc_error
    if IS_LIDAR:
        cmd = "{} -a {} -b {} -l 1".format(PCERROR_PATH, inputFile, outputFile)
    else:
        cmd = "{} -a {} -b {} -c 1".format(PCERROR_PATH, inputFile, outputFile)
    run_cmd(cmd, True)
