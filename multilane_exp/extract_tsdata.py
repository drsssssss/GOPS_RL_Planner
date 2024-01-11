
import os

#extract the tensorboard file and config file to target path with same floder structure



def extract_one_floder_data(source_path, target_path):
    for floders in os.listdir(source_path):
        if not os.path.exists(os.path.join(target_path, floders)):
            os.makedirs(os.path.join(target_path, floders))
        for files in os.listdir(os.path.join(source_path, floders)):
            if files.startswith("events.out.tfevents"):
                os.system("cp " + os.path.join(source_path, floders, files) + " " + os.path.join(target_path, floders, files))
            if files == "config.json":
                os.system("cp " + os.path.join(source_path, floders, files) + " " + os.path.join(target_path, floders, files))
            if files == 'apprfunc' and os.path.isdir(os.path.join(source_path, floders,'apprfunc')):
                os.makedirs(os.path.join(target_path, floders,'apprfunc'),exist_ok=True)
                for file in os.listdir(os.path.join(source_path, floders,files)):
                    if file.endswith('500000.pkl') or file.endswith('opt.pkl'):
                        os.system(
                            "cp " + os.path.join(source_path, floders, files, file) + " "
                            + os.path.join(target_path, floders, files, file))
    return 


def extract_data(source_path, target_path,alg_list, appr_list, floder_list=None):
    if floder_list == None:
        floder_list = []
        for alg in alg_list:
            for appr in appr_list:
                floder_list.append(alg.lower() + '_' + appr.lower())

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    for floder in floder_list:
        source_floder = os.path.join(source_path,floder)
        target_floder = os.path.join(target_path,floder)
        if not os.path.exists(target_floder):
            os.makedirs(target_floder)
        extract_one_floder_data(source_floder,target_floder)
    print("data extracted to {}".format(target_path))
    return 


if __name__ == "__main__":
    source_path ="/home/gaojiaxin/gops_carracing/GOPS/results/carracingraw"
    target_path = "/home/gaojiaxin/gops_carracing/GOPS/exp/temp"
    alg_list = ['dsac']
    appr_list = ['cnn']

    extract_data(source_path,target_path,alg_list=alg_list,appr_list=appr_list)
