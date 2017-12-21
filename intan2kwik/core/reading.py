import numpy as np
import logging
import glob
import os
import re

from intan2kwik.core.intan.util.read_header import read_header as read_header_rhd
from intan2kwik.core.intan_rhs.util.read_header import read_header as read_header_rhs

from intan2kwik.core.intan.load_intan import read_data as read_data_rhd
from intan2kwik.core.intan_rhs.load_intan import read_data as read_data_rhs

logger = logging.getLogger('intan2kwik.core.reading')
rh_search_string = '*.rh[d, s]'
r_num_from_str = re.compile("([a-zA-Z\-]+)([0-9]+)")

# a clunky wrapper for read_intan_header that selects the right version
def read_intan_header(file_path):
    """
    :param file_path (string): file path (ends with .rhs/.rhd)
    :return: header structure, corresponding to the rhs/rhd file specification
    """
    ext = os.path.split(file_path)[-1].split('.')[-1]
    with open(file_path, 'rb') as f:
        if ext == 'rhs':
            logger.debug('Entered RHS file')
            hdr = read_header_rhs(f)
        elif ext == 'rhd':
            logger.debug('Entered RHD file')
            hdr = read_header_rhd(f)
        else:
            raise Exeption('Unrecognized extension {}'.format(ext))
    return hdr


def read_data(file_path):
    """
    :param file_path (string): file path (ends with .rhs/.rhd)
    :return: header structure, corresponding to the rhs/rhd file specification
    """
    ext = os.path.split(file_path)[-1].split('.')[-1]
    if ext == 'rhs':
        logger.debug('Entered RHS file for reading data')
        data = read_data_rhs(file_path)
    elif ext == 'rhd':
        logger.debug('Entered RHD file for reading data')
        data = read_data_rhd(file_path)
    else:
        raise Exeption('Unrecognized extension {}'.format(ext))

    return data


def s_f_lookup(sec_name, intan_data):
    return intan_data['frequency_parameters']['{}_sample_rate'.format(sec_name)]


def native_to_board(native_name, names_dict=None):
    if names_dict is None:
        #short name of the channel: name of the group of channels in the header dictionary
        names_dict = {'adc': 'board_adc',
                      'din': 'board_dig_in',
                      'dac': 'board_dac',
                      'analog-in': 'board_adc',
                      'digital-in': 'board_dig_in',
                      'analog-out': 'board_dac'}

    name, number = r_num_from_str.match(native_name).groups()
    return names_dict[name[:-1]]


def native_to_t(native_name, names_dict=None):
    if names_dict is None:
        names_dict = {'adc': 't_board_adc',
                      'dac': 't_board_adc',
                      'din': 't_dig',
                      'analog-in': 't_board_adc',
                      'digital-in': 't_dig',
                      'analog-out': 't_board_dac'}
    name, number = r_num_from_str.match(native_name).groups()
    return names_dict[name[:-1]]


def chan_lookup(native_name, intan_data):

    board_section = native_to_board(native_name)
    logger.debug('Chan {} is in sec {}'.format(native_name, board_section))
    sec_key = board_section + '_channels'
    sec_meta = intan_data[sec_key]
    logger.debug('Sec meta {} is {}'.format(sec_key, sec_meta))

    sec_ch_names = [m['native_channel_name'].lower() for m in sec_meta]

    ch_found = [i for i, ch_name in enumerate(sec_ch_names) if ch_name == native_name]

    if not (len(ch_found) == 1):
        raise Exception("Did not find exactly one channel matching {}".format(native_name))

    chan_meta = sec_meta[ch_found[0]].copy()
    chan_meta['sec_name'] = board_section
    return chan_meta


def read_aux(native_name, intan_data, volt=True, intan_header=None):
    ch_meta = chan_lookup(native_name, intan_data)
    sec_name = ch_meta['sec_name']
    # read the channel from the corresponding data_setcion
    t_key = native_to_t(native_name)
    data_key = sec_name + '_data'
    ch_order = np.int(ch_meta['custom_order'])

    s_f = s_f_lookup(sec_name, intan_data)
    t = intan_data[t_key]

    x = intan_data[data_key][ch_order]

    if volt and 'adc' in sec_name:
        assert intan_header is not None, 'Need to give header to scale an aux channel'
        if intan_header['eval_board_mode'] == 1:
            x = np.multiply(152.59e-6, (x.astype(np.int32) - 32768))  # units = volts
        else:
            x = np.multiply(50.354e-6, x)

    return x, s_f, t


def read_intan_rec(rec_folder):
    logger.info('reading intan rec {}'.format(rec_folder))
    # list all the chunk files:
    all_rhx_files = glob.glob(os.path.join(rec_folder, rh_search_string))
    all_rhx_files.sort()
    logger.info('Found {} chunks'.format(len(all_rhx_files)))

    all_intan_rec = []

    for i_file, rhd_file in (enumerate(all_rhx_files)):
        all_intan_rec.append(read_data(rhd_file))

    return all_intan_rec


def read_aux_all_rec(native_name_list, rec_folder, volt=True):
    # go through all the records extracting the one channel (very inefficient)
    # it reads all the block for the record
    logger.info('reading intan chans {0} across all of rec {1}'.format(native_name_list, rec_folder))
    all_rhd_files = glob.glob(os.path.join(rec_folder, rh_search_string))
    all_rhd_files.sort()
    logger.info(all_rhd_files)
    # list all the chunk files:
    first_header = read_intan_header(all_rhd_files[0])
    all_intan_rec = []

    chan_dictionaries = [{'name': n, 'x': np.array([]), 't': np.array([])} for n in native_name_list]

    for i_file, rhd_file in (enumerate(all_rhd_files)):
        logger.info('file {}/{}'.format(i_file, len(all_rhd_files)))
        block_read = read_data(rhd_file)

        for one_chan_dict in chan_dictionaries:
            native_name = one_chan_dict['name']
            x, s_f, t = read_aux(native_name, block_read, volt=volt, intan_header=first_header)
            one_chan_dict['t'] = np.hstack([one_chan_dict['t'], t])
            one_chan_dict['x'] = np.hstack([one_chan_dict['x'], x])

            if i_file == 0:
                one_chan_dict['meta'] = chan_lookup(native_name, block_read)
                one_chan_dict['s_f'] = s_f

    return chan_dictionaries









