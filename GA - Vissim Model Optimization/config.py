def run_config():

    # ---------------------------------------------------
    # Simulation Configuration
    # ---------------------------------------------------

    # Number of runs per iteration
    run = 6
    # Simulation period per run
    period = 4600
    
    dic = {'run':run, 'period':period, }
    return(dic)

def db_config():

    # ------------------------------------------------------
    # Driving Behavior Config
    # ------------------------------------------------------

    # following parameters
    cc0_min = 1
    cc0_max = 10
    cc1_min = 1
    cc1_max = 3
    cc4_min = -1
    cc4_max = 0
    cc5_min = 0
    cc5_max = 1

    # number of bins for following parameters
    cc0_incr = 0.611911872
    cc1_incr = 1
    cc4_incr = 0.228
    cc5_incr = 0.228

    # Maximum cooperative deceleration
    mcd_min = 0
    mcd_max = 10
    # Cooperative lane change - maximum speed difference
    msd_min = 10
    msd_max = 20
    # Maximum deceleration (own/trailing veh)
    odrd_min = -1
    odrd_max = -10
    tdrd_min = -1
    tdrd_max = -10
    # Accepted deceleration (own/trailing veh)
    oad_min = -1
    oad_max = -10
    tad_min = -1
    tad_max = -10
    # Safety distance reduction factor
    sdrf_min = 0.1
    sdrf_max = 1

    # Increment for lane change parameters
    mcd_incr = 0.1
    msd_incr = 0.01
    odrd_incr = 0.1
    tdrd_incr = 0.1
    oad_incr = 0.1
    tad_incr = 0.1
    sdrf_incr = 0.05

    dic = {'cc0_min': cc0_min, 'cc0_max': cc0_max, 'cc1_min': cc1_min, 'cc1_max': cc1_max,
           'cc4_min': cc4_min, 'cc4_max': cc4_max, 'cc5_min': cc5_min, 'cc5_max': cc5_max,
           'mcd_min': mcd_min, 'mcd_max': mcd_max, 'msd_min': msd_min, 'msd_max': msd_max,
           'odrd_min': odrd_min, 'odrd_max': odrd_max, 'tdrd_min': tdrd_min, 'tdrd_max': tdrd_max,
           'oad_min': oad_min, 'oad_max': oad_max, 'tad_min': tad_min, 'tad_max': tad_max,
           'sdrf_min': sdrf_min, 'sdrf_max': sdrf_max, 'cc0_incr': cc0_incr, 'cc1_incr': cc1_incr,
           'cc4_incr': cc4_incr, 'cc5_incr': cc5_incr, 'mcd_incr':mcd_incr,'msd_incr':msd_incr,
           'odrd_incr':odrd_incr,'tdrd_incr':tdrd_incr,'oad_incr':oad_incr,'tad_incr':tad_incr,
           'sdrf_incr':sdrf_incr}
    return(dic)

def threshold():

    # ---------------------------------------------------
    # Travel Time Diff Threshold for Each Route (seconds)
    # ---------------------------------------------------

    # 1: 152 to PMH-mid
    t1 = 0
    # 2: PMH-mid to CapeHorn
    t2 = 0
    # 3: CapeHorn to King Edwards
    t3 = 0
    # 4: King Edwards to West Brunette
    t4 = 0
    # 5: Gaglardi to Burnaby Lake
    t5 = 0
    # 6: Burnaby Lake to Kensington
    t6 = 0
    # 7: Kensington to West Kensington
    t7 = 0
    # 8: West Kensington to Willingdon
    t8 = 0
    # 9: Willingdon to First Ave
    t9 = 0
    # 10: First Ave to Hastings
    t10 = 5

    dic = {'t1':t1, 't2':t2, 't3':t3, 't4':t4, 't5':t5, 't6':t6,
           't7':t7, 't8':t8, 't9':t9, 't10':t10}
    return(dic)
