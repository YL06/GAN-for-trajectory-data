import webbrowser
from transform import transfer
def draw_traj(trajs, output_file_name = 'map.html', if_append = False):
    mode = 'a' if if_append else 'w'
    # added encoding
    with open('template/template.html', 'r', encoding='utf-8') as f1, open(output_file_name, mode) as f2:
        read_lines = f1.readlines() # 整行读取数据 Read the entire line of data
        write = ''
        for line in read_lines:
            l = line.strip()
            if l == '//add here':
                write += '//add here\n'
                for traj in trajs:
                    write += '\tvar pois = [\n'
                    for pos in traj:
                        # ('+str(pos[1])+','+str(pos[0])+') changed
                        write = write + '\t\tnew BMap.Point('+str(pos[0])+','+str(pos[1])+'),\n'
                    write = write[:-2]+'\n\t];\n\ttraj.push(pois);\n'
            else:
                write += line

        f2.write(write)
        f1.close()
        f2.close()
    webbrowser.open(output_file_name)

if __name__ == '__main__':
    import numpy as np
    tr = transfer()
    #d = np.load('D:/YJ/06_05/119.npy', allow_pickle=True)
    #     d = np.load('E:/data/ori_trj_data/real_training_data_10000/9212.npy', allow_pickle=True)
    #     e = np.load('D:/YJ/06_05/112.npy', allow_pickle=True)
    d = np.load('E:/different_sizes/Connect/generated_from_1000/101.npy', allow_pickle=True)
    e = np.load('E:/different_sizes/Connect/generated_from_1000/100.npy', allow_pickle=True)
    #for i in range(len(d)):
    #    d[i] = [tr.wg84_to_bd09(float(d[i,1]),float(d[i,0]))[1],tr.wg84_to_bd09(float(d[i,1]),float(d[i,0]))[0]]
    #for i in range(len(e)):
    #    e[i] = [tr.wg84_to_bd09(float(e[i,0]),float(e[i,0]))[1],tr.wg84_to_bd09(float(e[i,1]),float(e[i,0]))[0]]

    draw_traj([d,e])
