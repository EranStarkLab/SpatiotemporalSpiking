import numpy as np
import time
import scipy.io

from data_utils.clusters import Spike, Cluster
from data_utils.xml_reader import read_xml
from data_utils.light_removal import remove_light
from constants import NUM_CHANNELS, TIMESTEPS, NUM_BYTES, SESSION_EMPTY_INDS

from paths import DATA_PATH, DATA_MAT_PATH, DATA_TEMP_PATH, TEMP_PATH_FULL


def get_next_spike(spk_file):
    """
    input:
    spk_file: file descriptor; the file from which to read

    return:
    spike: Spike object; containing the next spike in the file
    """
    data = np.zeros((NUM_CHANNELS, TIMESTEPS))
    for i in range(TIMESTEPS):
        for j in range(NUM_CHANNELS):
            num = spk_file.read(NUM_BYTES)
            if not num:
                return None

            data[j, i] = int.from_bytes(num, "little", signed=True)
    spike = Spike()
    spike.data = data
    return spike


def get_next_time(res_file):
    """
    input:
    res_file: file descriptor; the file from which to read

    return:
    timing: int; the timing of the next spike
    """
    timing = res_file.readline()
    try:
        timing = int(timing)
    except ValueError:
        timing = None

    return timing


def get_next_cluster_num(clu_file):
    """
    input:
    clu_file: file descriptor; the file from which to read

    return:
    num: int; cluster ID
    """
    num = clu_file.readline()
    assert num != ''
    return int(num)


def find_indices_in_filenames(target_name, cell_class_mat):
    """
    Finds the relevant slice in the tagging information

    input:
    target_name: string; recording name
    cell_class_mat: list; spv information

    return:
    start_index, end_index: integer tuple; start and end indices of the relevant data in the tagging mat
    """
    file_name_arr = cell_class_mat['filename']

    index = 0
    start_index = 0
    # find first occurrence of target_name
    for filenameArr in file_name_arr:
        filename = filenameArr
        if filename == target_name:
            start_index = index
            break
        index += 1

    # find last occurrence of target_name
    for i in range(start_index, len(file_name_arr)):
        if file_name_arr[i] != target_name:
            return start_index, i

    end_index = len(file_name_arr)

    return start_index, end_index


def find_cluster_index_in_shankclu_vector(start_index, end_index, shank_num, clu_num, cell_class_mat):
    """
    Finds the index in the spv for the data

    input:
    start_index: int; start of relevant data
    end_index: int; end of relevant data
    shank_num: int shank number
    clu_num: int; cluster ID
    cell_class_mat: list; tagging information

    return:
    index: integer; relevant index in the tagging information, None if not found
    """
    shank_clu_vec = cell_class_mat['shankclu']
    for i in range(start_index, end_index):
        shank_clu_entry = shank_clu_vec[i]
        if shank_clu_entry[0] == shank_num and shank_clu_entry[1] == clu_num:  # found
            return i
    return None


def determine_cluster_label(filename, shank_num, clu_num, cell_class_mat):
    """
    Determines cluster's label based on the tagging information

    input:
    filename: string; recording name
    shank_num: int; shank number
    clu_num: int; cluster ID
    cell_class_mat: list; tagging information

    return:
    label: integer; see function's body for specification
    """
    start_index, end_index = find_indices_in_filenames(filename, cell_class_mat)
    clu_index = find_cluster_index_in_shankclu_vector(start_index, end_index, shank_num, clu_num, cell_class_mat)
    is_act = cell_class_mat['act'][clu_index]
    is_exc = cell_class_mat['exc'][clu_index]
    is_inh = cell_class_mat['inh'][clu_index]

    if clu_index is None:
        return -2

    # 0 = PV
    # 1 = Pyramidal
    # -3 = both (pyr and PV) which means it will be discarded
    # -1 = untagged
    # -2 = clusters that appear in clu file but not in shankclu
    if is_exc == 1:
        if is_act == 1 or is_inh == 1:  # check if both conditions apply (will be discarded)
            return -3

        return 1

    if is_act == 1 or is_inh == 1:
        return 0
    return -1


def create_cluster(name, clu_num, shank_num, cell_class_mat):
    """
    input:
    name: string; recording name
    clu_num: integer; cluster ID
    shank_num: integer; shank number
    cell_class_mat: list; containing the tagging information

    return:
    cluster: new Cluster object
    """
    # get cluster's label
    label = determine_cluster_label(name, shank_num, clu_num, cell_class_mat)

    # Check if the cluster doesn't appear in shankclu
    if label == -2:
        return None

    cluster = Cluster(label=label, filename=name, num_within_file=clu_num, shank=shank_num)
    return cluster


def read_directory(path, cell_class_mat, i):
    """
    The reader function.

    input:
    path: string; path to the to the recording directory
    cell_class_mat: list; the tagging information
    i: integer; the shank number

    return:
    clusters_list: list of Cluster objects
    """
    clusters = dict()
    clusters_list = []
    name = path.split("/")[-1]

    start = time.time()
    try:
        spk_file = open(path + "/" + name + ".spk." + str(i), 'rb')  # file containing spikes
        clu_file = open(path + "/" + name + ".clu." + str(i))  # file containing cluster mapping of spikes
        res_file = open(path + "/" + name + ".res." + str(i))  # file containing spike timings
    except FileNotFoundError:  # if shank recording doesn't exsist exit
        print(path + "/" + name + ".spk." + str(i) +
              ' and/or ' + path + "/" + name + ".clu." + str(i) +
              'and/or ' + path + "/" + name + ".res." + str(i) + ' not found')
        return []

    # Read the first line of the cluster file (contains num of clusters)
    get_next_cluster_num(clu_file)
    spike = get_next_spike(spk_file)
    timing = get_next_time(res_file)
    while spike is not None:  # for each spike
        assert timing is not None
        clu_num = get_next_cluster_num(clu_file)  # cluster ID

        # clusters 0 and 1 are artefacts and noise by convention
        if clu_num == 0 or clu_num == 1:
            spike = get_next_spike(spk_file)
            timing = get_next_time(res_file)
            continue

        assert clu_num is not None
        full_name = name + "_" + str(i) + "_" + str(clu_num)  # the format is filename_shankNum_clusterNum

        # Check if cluster exists in dictionary and create if not
        if full_name not in clusters:
            new_cluster = create_cluster(name, clu_num, i, cell_class_mat)

            # Check to see if the cluster we are trying to create is one that doesn't appear in shankclu (i.e has a
            # label of -2)
            if new_cluster is None:
                spike = get_next_spike(spk_file)
                timing = get_next_time(res_file)
                continue

            clusters[full_name] = new_cluster
            clusters_list.append(new_cluster)

        clusters[full_name].add_spike(spike, timing)
        spike = get_next_spike(spk_file)
        timing = get_next_time(res_file)

    print("Finished File %s with index %d" % (name, i))
    spk_file.close()
    clu_file.close()

    end = time.time()
    print(str(end - start) + " total")

    return clusters_list


def read_all_directories(path_to_mat, groups=None):
    """
    Generator function to iterate over session directories

    input:
    path_to_mat: string; path to the mat file containing the tagging information

    return:
    dir_clusters: list of Cluster objects; each time all the clusters from a single shank from a single recording
    """
    cell_class_mat = scipy.io.loadmat(path_to_mat, simplify_cells=True)
    for key in SESSION_EMPTY_INDS:
        data_dir, remove_inds = DATA_PATH + key, SESSION_EMPTY_INDS[key]
        print("reading " + str(data_dir))
        for i in range(1, 5):
            if str(i) in remove_inds:  # skip shanks according to instruction in dirs file
                print('Skipped shank %d in file %s' % (i, data_dir))
                continue
            name = key + f"_{i}"
            if groups is not None and groups[name] != NUM_CHANNELS:
                print('Skipped shank %d in file %s according to xml' % (i, data_dir))
                continue
            dir_clusters = read_directory(data_dir, cell_class_mat, i)  # read the data files of shank i
            print("the number of clusters is: " + str(len(dir_clusters)))
            yield dir_clusters

def main():
    groups = read_xml(DATA_PATH)

    clusters_generator = read_all_directories(DATA_MAT_PATH, groups)

    for clusters in clusters_generator:
        for cluster in clusters:  # for each unit
            cluster.fix_punits()

            cluster.save_cluster(TEMP_PATH_FULL)

            inds = remove_light(cluster, True, DATA_PATH)
            cluster.timings = cluster.timings[inds]
            cluster.finalize_cluster()
            cluster.np_spikes = cluster.np_spikes[inds]

            cluster.save_cluster(DATA_TEMP_PATH)


if __name__ == "__main__":
    main()
