
def get_channel_setting(sid):
    if sid==1:
        chn_num=46
        active_chn=[0, 3, 6, 12, 13, 14, 19, 20, 21, 22, 23, 24, 27, 34, 35, 36, 37, 44, 45]
    else:
        print("No participant ID found.")
    return active_chn

