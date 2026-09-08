def cnfg_mock():
    
    listisec = [1]
    listicam = [4]
    listiccd = [1]
    main( \
         listisec=listisec, \
         listicam=listicam, \
         listiccd=listiccd, \
         datatype='mock')


def cnfg_mich():

    pathdata = tdpy.ensr_path(os.path.join(tdpy.retr_pathbase('tpet'), 'data', 'tesscut'))
    listpath = fnmatch.filter(os.listdir(pathdata), 'tess*')
    for p in range(len(listpath)):
        isec = int(listpath[p][6:10])
        icam = int(listpath[p][11])
        iccd = int(listpath[p][13])
        pathfile = os.path.join(pathdata, listpath[p])
        if isec == 7 or isec == 8:
            main(isec, icam, iccd, pathfile=pathfile)


def cnfg_obsd():
    
    listisec = [16]
    listicam = [1]
    listiccd = [1]
    main( \
         listisec=listisec, \
         listicam=listicam, \
         listiccd=listiccd, \
         datatype='mock', \
         strgmode='tcut', \
         rasctarg=105.04811, \
         decltarg=-66.04004, \

         #numbside=256, \
         #rasctarg=7.0032072, \
         #decltarg=-66.0400389, \
         )



globals().get(sys.argv[1])()


