cat_list = ['ip', 'app', 'device', 'os', 'channel', 
    'mobile', 'mobile_app', 'mobile_channel', 'app_channel'
    ]
datatype_list = {
    'ip'                : 'uint32',
    'app'               : 'uint16',
    'device'            : 'uint16',
    'os'                : 'uint16',
    'channel'           : 'uint16',
    'is_attributed'     : 'uint8',
    'click_id'          : 'uint32',
    'mobile'            : 'uint16',
    'mobile_app'        : 'uint16',
    'mobile_channel'    : 'uint16',
    'app_channel'       : 'uint16',
    'category'          : 'category',
    'epochtime'         : 'int64',
    'nextClick'         : 'int64',
    'nextClick_shift'   : 'float64',
    'min'               : 'uint8',
    'day'               : 'uint8',
    'hour'              : 'uint8',
    'ip_mobile_day_count_hour'              : 'uint32',
    'ip_mobileapp_day_count_hour'           : 'uint32',
    'ip_mobilechannel_day_count_hour'       : 'uint32',
    'ip_appchannel_day_count_hour'          : 'uint32',
    'ip_mobile_app_channel_day_count_hour'  : 'uint32'
    }

CAT_COMBINATION_FILENAME = 'cat_combination.csv'
NEXTCLICK_FILENAME = 'nextClick.csv'
TIME_FILENAME = 'day_hour_min.csv'

nchunks = 10000000    
nchunks = 100000