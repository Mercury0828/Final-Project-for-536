
import numpy as np
import glob

if __name__ == '__main__':
    configure_list = ['center_single', 'in_center_single_out_center_single',
                      'left_center_single_right_center_single',
                      'up_center_single_down_center_single']
    for configure in configure_list:
        print(configure)
        for mode in ['train', 'valid', 'test']:
            print(' ' * 3, mode)
            mode_str = 'val' if mode == 'valid' else mode
            npz_list = glob.glob(f'./Dataset/{configure}/*_{mode_str}.npz')
            image = []
            target = []
            meta_target = []
            structure = []
            for i, npz_path in enumerate(npz_list):
                npz = np.load(npz_path)
                image.append(npz['image'])
                target.append(npz['target'])
                meta_target.append(npz['meta_target'])
                structure.append(npz['structure'])
                if (i+1) % 1000 == 0:
                    np.savez(f'./Dataset/{configure}_{mode}_{(i+1)//1000}.npz',
                             image=np.stack(image), target=np.array(target),
                             meta_target=np.stack(meta_target),
                             structure=np.stack(structure))
                    image = []
                    target = []
                    meta_target = []
                    structure = []
            if len(target) > 0:
                np.savez(f'./Dataset/{configure}_{mode}_{(i+1)//1000 + 1}.npz',
                         image=np.stack(image), target=np.array(target),
                         meta_target=np.stack(meta_target),
                         structure=np.stack(structure))
                image = []
                target = []
                meta_target = []
                structure = []
