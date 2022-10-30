import xml.etree.ElementTree as ET
from pathlib import Path

def read_xml(path):
    """

    Parameters
    ----------
    path : String; path to where the xml describing the recording process can be found

    Returns
    -------
    Dictionary coupling the session and shank to the number of recording channels
    """
    ret = {}

    path_list = Path(path).rglob('*.xml')

    for xml_path in path_list:
        xml_path = str(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        spikes = root.find('spikeDetection')
        groups = spikes.find('channelGroups').findall('group')
        file_name = '.'.join(xml_path.split('\\')[-1].split('.')[:-1])

        for i, group in enumerate(groups):
            channels = group.find('channels')
            num_channels = len(channels.findall('channel'))
            ret[f"{file_name}_{i + 1}"] = num_channels

    return ret
