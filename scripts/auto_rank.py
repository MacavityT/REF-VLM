import os
import re
import sys
import time

def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|') #根据nvidia-smi命令的返回值按照'|'为分隔符建立一个列表
    '''
    结果如：
    ['', ' N/A   64C    P0    68W /  70W ', '   9959MiB / 15079MiB ', '     79%      Default ', 
    '\n', ' N/A   73C    P0   108W /  70W ', '  11055MiB / 15079MiB ', '     63%      Default ', 
    '\n', ' N/A   60C    P0    55W /  70W ', '   3243MiB / 15079MiB ', '     63%      Default ', '\n']
    '''
    gpus_num = len(gpu_status) // 4
    gpus_memory = []
    gpus_power = []
    gpus_util = []
    for i in range(gpus_num):
        base = 4*i + 1
        #获取GPU当前显存使用量
        gpu_power = int(gpu_status[base].split('   ')[-1].split('/')[0].split('W')[0].strip())
        #获取GPU功率值：提取标签为2的元素，按照'/'为分隔符后提取标签为0的元素值再按照'M'为分隔符提取标签为0的元素值，返回值为int形式 
        gpu_memory = int(gpu_status[base + 1].split('/')[0].split('M')[0].strip()) 
        #获取GPU显存核心利用率
        # gpu_util = int(gpu_status[base+2].split('   ')[1].split('%')[0].strip())
        gpu_util = int(re.search(r'\d+%', gpu_status[base+2]).group().split('%')[0].strip())

        gpus_memory.append(gpu_memory)
        gpus_power.append(gpu_power)
        gpus_util.append(gpu_util)
        
    mean_memory = sum(gpus_memory) / gpus_num
    mean_power = sum(gpus_power) / gpus_num
    mean_util = sum(gpus_util) / gpus_num
    
    return mean_power, mean_memory, mean_util
 
def narrow_setup(cmd, secs=600):  #间隔十分钟检测一次
    gpu_power, gpu_memory, gpu_util = gpu_info()
    i = 0
    cnt = 0
    while not(gpu_memory < 100 and gpu_power < 100 and gpu_util < 10 and cnt > 1) :  # 当两次检测中平均功率，使用量，利用率都小于特定值才去退出循环
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        gpu_util_str = 'gpu util:%d %% |' % gpu_util
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + gpu_util_str + ' ' + symbol)
        #sys.stdout.write(obj+'\n')等价于print(obj)
        sys.stdout.flush()    #刷新输出
        time.sleep(secs)  #推迟调用线程的运行，通过参数指秒数，表示进程挂起的时间。
        i += 1

        # 监视信息
        gpu_power, gpu_memory, gpu_util = gpu_info()
        if gpu_memory < 100 and gpu_power < 100 and gpu_util < 10: 
            cnt += 1
        else:
            cnt = 0 # reset cnt

    print('\n' + cmd)
    os.system(cmd) #执行脚本
 
 
if __name__ == '__main__':
    import argparse
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-c', '--cmd', default='bash scripts/train.sh', type=str, help='command to run')
    argparse.add_argument('-s', '--secs', default=600, type=int, help='interval time')
    args = argparse.parse_args()
    narrow_setup(args.cmd, args.secs)