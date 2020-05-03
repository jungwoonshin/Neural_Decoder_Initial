from scipy import stats
import numpy as np

with open('mean_roc_and_ap_with_feature', 'r') as filehandle:
    filecontents = filehandle.readlines()
    roc_ap = []
    for i in filecontents:
    	roc_ap.append(float(i.strip('\n')))
    roc_with_feat = roc_ap[0:20]
    ap_with_feat = roc_ap[20:]
    print(len(roc_with_feat))
    print(len(ap_with_feat))
    
with open('mean_roc_and_ap', 'r') as filehandle:
    filecontents = filehandle.readlines()
    roc_ap = []
    for i in filecontents:
    	roc_ap.append(float(i.strip('\n')))
    roc_without_feat = roc_ap[0:20]
    ap_without_feat = roc_ap[20:]
    print(len(roc_without_feat))
    print(len(ap_without_feat))

t_val, p_val = stats.ttest_ind(roc_without_feat,roc_with_feat,equal_var=False)
print(t_val, p_val)
print('roc t_val=','{:.5f}'.format(t_val),', roc p_val=','{:.5f}'.format(p_val))

t_val, p_val = stats.ttest_ind(ap_with_feat,ap_without_feat,equal_var=False)
print('ap t_val=','{:.5f}'.format(t_val),', ap p_val=','{:.5f}'.format(p_val))

mean_roc, std_roc = np.mean(roc_with_feat), np.std(roc_with_feat)
mean_ap, std_ap = np.mean(ap_with_feat), np.std(ap_with_feat)

print('mean_roc_with_feat=','{:.5f}'.format(mean_roc),', std_roc_with_feat=','{:.5f}'.format(std_roc))
print('mean_ap_with_feat=','{:.5f}'.format(mean_ap),', std_ap_with_feat=','{:.5f}'.format(std_ap))

mean_roc, std_roc = np.mean(roc_without_feat), np.std(roc_without_feat)
mean_ap, std_ap = np.mean(ap_without_feat), np.std(ap_without_feat)

print('mean_roc_without_feat=','{:.5f}'.format(mean_roc),', std_roc_without_feat=','{:.5f}'.format(std_roc))
print('mean_ap_without_feat=','{:.5f}'.format(mean_ap),', std_ap_without_feat=','{:.5f}'.format(std_ap))