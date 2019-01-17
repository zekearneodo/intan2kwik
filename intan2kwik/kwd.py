import logging
import numpy as np
import glob
import os
import h5py
import warnings
import tempfile
import shutil
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import contextlib

from intan2kwik.core.h5 import tables
from intan2kwik.core import reading
from intan2kwik.core.file import util as fu
#from intan2kwik.core.reading import rh_search_string

logger = logging.getLogger('intan2kwik.kwd')


def h5_unicode_hack(x):
    if isinstance(x, str):
        x = x.encode('utf8')
    return x


def list_chan_names(header, include_channels):
    ch_groups = [c + '_channels' for c in include_channels]

    c_names = []
    for chgrp in ch_groups:
        ch_list = header[chgrp]
        c_names += [ch['custom_channel_name'] for ch in ch_list]
    return c_names


def list_chan_bit_volts(header, include_channels):
    v_multipliers = [0.195, 50.354e-6]
    if header['eval_board_mode'] == 1:
        v_multipliers[1] = 152.59e-6

    ch_groups = [c + '_channels' for c in include_channels]
    b_volts = []

    for chgrp in ch_groups:
        ch_list = header[chgrp]
        if 'adc' in chgrp:
            b_value = v_multipliers[1]
        elif 'amplifier' in chgrp:
            b_value = v_multipliers[0]
        b_volts += [b_value for ch in ch_list]

    return np.array(b_volts, dtype=np.float)


def rhd_data_block(rhd_file, include_chans, times_too=False):
    read_block = reading.read_data(rhd_file)
    # identify channels, make numy array with all the block and a list of metadata for each
    block_data = np.vstack([read_block['{}_data'.format(ch_grp)]
                            for ch_grp in include_chans])
    block_t = np.vstack([read_block['t_{}'.format(ch_grp)]
                         for ch_grp in include_chans])

    if times_too:
        ret_value = (block_data, block_t)
    else:
        ret_value = block_data

    return ret_value


def rhd_rec_to_table(rhd_list: list, parent_group: h5py.Group, chan_groups_wishlist) -> np.array:
    '''
    :param rhd_list: (list of strings) paths of files to include in this rec
    :param parent_group: (h5 object group) parent group for this rec
    :param chan_groups_wishlist: (flat ndarray/list) channel groups to get from the rhd files
    :return: end_sample_vec: nd.array(2, n_rec)
    '''
    # make the table
    # read the blocks and append them to the table
    # set the attributes of the table
    logger.debug('Appending {} files to data table in {}'.format(len(rhd_list),
                                                                 parent_group))

    total_samples_in = 0
    total_samples_in_dig = 0

    last_t = 0
    # keep track of the end t of each file (t, t_dig)
    end_sample_vec = np.zeros([2, len(rhd_list)], dtype=np.int64)

    s_f = parent_group.attrs.get('sample_rate')

    rec_id = parent_group.name.split('/')[-1]

    # make a cute progress bar
    pbar = tqdm(enumerate(rhd_list), total=len(rhd_list),
                            desc='rec {}'.format(rec_id), leave=False)
    for i, rhd_file in pbar:
        logger.debug('Reading file {0}/{1}'.format(i+1, len(rhd_list)))
        pbar.set_postfix(file=' {}'.format(os.path.split(rhd_file)[-1]))
        read_block = reading.read_data(rhd_file)
        if i == 0:
            # filter include groups, warn if a group wasn't in the data and remove it from the list
            include_chan_groups = []
            for g in chan_groups_wishlist:
                if '{}_data'.format(g) in read_block.keys():
                    include_chan_groups.append(g)
                else:
                    logger.warn(
                        'Channel data group {} is not in the recordings'.format(g))
        # The main data (neural chans and dac chans; from include_chans)
        block_data = np.vstack([read_block['{}_data'.format(ch_grp)]
                                for ch_grp in include_chan_groups])
        block_t = np.vstack([read_block['t_{}'.format(ch_grp)]
                             for ch_grp in include_chan_groups])
        save_block = block_data.T.astype(np.int32) - 32768
        # assuming the first block in include_chans is amplifier
        save_t = block_t[0].reshape([-1, 1])
        name_t = 't_{}'.format(include_chan_groups[0])

        # The digital channels
        try:
            dig_in_data = read_block['board_dig_in_data'].T.astype(np.short)
            dig_in_t = read_block['t_dig'].reshape([-1, 1])
            has_digital_in = True
        except KeyError as k:
            logger.warn('No digital channels')
            has_digital_in = False

        # data comes in as uint16
        if i == 0:
            logger.debug('Creating tables of neural/adc data')
            dset = tables.unlimited_rows_data(parent_group, 'data',
                                              save_block.astype(np.int16))
            tset = tables.unlimited_rows_data(parent_group, name_t, save_t)

            if has_digital_in:
                logger.debug('Creating tables of digital data')
                dset_dig = tables.unlimited_rows_data(parent_group, 'dig_in',
                                                      dig_in_data.astype(np.short))
                tset_dig = tables.unlimited_rows_data(
                    parent_group, 't_dig', dig_in_t)

        else:
            tables.append_rows(dset, save_block.astype(np.int16))
            tables.append_rows(tset, save_t)

            if has_digital_in:
                tables.append_rows(dset_dig, dig_in_data.astype(np.short))
                tables.append_rows(tset_dig, dig_in_t)

            # assert time continuity
            more_control_d_samples = (save_t[0] * s_f - total_samples_in)
            logger.debug(
                'Delta cum_samples/cum_t is {}'.format(more_control_d_samples))

            control_dt = save_t[0] - last_t
            control_delta_samples = int(np.round(control_dt * s_f))
            # logger.info('Delta samples between rhd files is {}'.format(control_delta_samples))
            if not control_delta_samples == 1:
                warn_msg = 'Skipped a beat i rhd files diff is {}s'.format(
                    control_dt)
                warnings.warn(warn_msg, RuntimeWarning)

        last_t = save_t[-1]
        total_samples_in += save_block.shape[0]
        end_sample_vec[0, i] = total_samples_in - 1

        if has_digital_in:
            last_t_dig = dig_in_t[-1]
            total_samples_in_dig += last_t_dig.shape[0]
            end_sample_vec[1, i] = total_samples_in_dig

    # only atrribute for table is valid_samples
    dset.attrs.create('valid_samples', np.ones(
        save_block.shape[1]) * total_samples_in)

    if has_digital_in:
        dset_dig.attrs.create('valid_samples', np.ones(
            dig_in_data.shape[1]) * total_samples_in_dig)

    return end_sample_vec


def create_data_grp(rec_grp, intan_hdr, include_channels, rec_rhx_pd):
    rec = rec_rhx_pd['rec'].values[0]
    logger.debug('Creating data group for this rec {}'.format(rec))
    rec_start = rec_rhx_pd.iloc[0, :]['t_stamp'].strftime("%Y-%m-%d %H:%M:%S").encode('utf-8')

    data_grp = rec_grp.create_group('{}'.format(rec))
    # append the metadata to this data group
    data_grp.attrs.create('bit_depth', 16, dtype=np.uint32)
    data_grp.attrs.create('sample_rate', intan_hdr['sample_rate'], dtype=np.float32)
    data_grp.attrs.create('name', rec)
    data_grp.attrs.create('start_sample', 0, dtype=np.uint32)
    data_grp.attrs.create('start_time', rec_start)

    all_chan_names = list_chan_names(intan_hdr, include_channels)
    all_chan_names_uni = [h5_unicode_hack(x) for x in all_chan_names]
    all_bit_volts = list_chan_bit_volts(intan_hdr, include_channels)
    n_chan = len(all_chan_names)
    all_rates = np.ones(n_chan) * intan_hdr['sample_rate']

    # create application data
    app_data_grp = data_grp.create_group('application_data')
    app_data_grp.attrs.create('is_multiSampleRate_data', 0, dtype=np.uint8)
    app_data_grp.attrs.create('channels_sample_rate', all_rates)
    app_data_grp.attrs.create('channel_bit_volts', all_bit_volts)
    app_data_grp.attrs.create('channel_names', all_chan_names_uni)

    return data_grp


def which_board(folder):
    rhx_files = glob.glob(os.path.join(folder, '*.rh[d,s]'))
    if len(rhx_files) < 1:
        raise RuntimeError('No .rh[d, x] found in folder {}'.format(folder))
    return os.path.split(rhx_files[0])[-1].split('.')[-1]


def neural_ports_meta(folder):
    """
    :param folder: (string) folder with .rh? files
    :return: (tuple) of dictionaries:
        chans: {port: list of channels}
        count: {port: channel count
    """
    board = which_board(folder)
    rec_rhx_files = glob.glob(os.path.join(folder, '*.{}'.format(board)))
    rec_rhx_files.sort()
    intan_read = reading.read_data(rec_rhx_files[0])
    chans, count = used_neural_channels(intan_read)
    return chans, count


def intan_to_kwd_multirec(kwd_file, sess_pd: pd.DataFrame, include_channels=None):
    # make the .kwd file
    # make the /recording/{} i group
    # dump header to application data
    # run rhd_rec_to table to create the table in the group

    folder = os.path.split(sess_pd.loc[0, 'path'])[0]
    if include_channels is None:
        include_channels = ['amplifier', 'board_adc']
    logger.debug(
        'reading intan chans data across all of sess {0}'.format(folder))

    # list all the chunk files:
    rec_grp = kwd_file.require_group('recordings')
    all_recs = np.unique(sess_pd['rec'])
    logger.debug('File should have {} recs'.format(all_recs.shape[0]))
    for rec in tqdm(all_recs, total=len(all_recs), desc='Sess'):
        rec_rhx_pd = sess_pd[sess_pd['rec'] == rec]
        rec_rhx_files = list(sess_pd[sess_pd['rec'] == rec]['path'])
        # attributes from the header
        first_header = reading.read_intan_header(rec_rhx_files[0])
        # logger.info('First header {}'.format(first_header))

        v_multipliers = [0.195, 50.354e-6]
        if first_header['eval_board_mode'] == 1:
            v_multipliers[1] = 152.59e-6

        logger.debug(
            'Creating the /recordings/{} with {} files'.format(rec, len(rec_rhx_files)))
        data_grp = create_data_grp(
            rec_grp, first_header, include_channels, rec_rhx_pd)
        # create application data with all metadata
        logger.debug('Creating data table and going throug the recs')

        timestamp_recs = rhd_rec_to_table(
            rec_rhx_files, data_grp, include_channels)
        # create the table of files, and the tabe with rec end timestamps
        table_names = ['rhx_files', 't_stamp_end', 't_stamp_dig_end']
        table_vectors = [np.array(rec_rhx_files, dtype=h5py.special_dtype(vlen=str)),
                            timestamp_recs[0],
                            timestamp_recs[1]]
        for name, vec in zip(table_names, table_vectors):
            tables.insert_table(data_grp['application_data'], vec, name)

    return first_header


def intan_to_kwd(folder, dest_file_path, rec=0, include_channels=None, board='auto', single_rec=False):
    """
    :param folder: (string) folder where the .rh? files are
    :param dest_file_path: (string) dest of the kw? files
    :param rec: (int)
    :param include_channels: (flat ndarray/list)
    :param board: (str) 'rhd' or 'rhs' for rhd2000 or rhs2000, 'auto' for detecting from extension
    :param sigle_rec: bool, whether to lump everything into one recording, or try to split each continuous segment
    :return:
    """
    # make the .kwd file
    # make the /recording/0 group
    # dump header to application data
    # run rhd_rec_to table to create the table in the group
    if include_channels is None:
        include_channels = ['amplifier', 'board_adc']
    logger.info(
        'reading intan chans data across all of rec {0}'.format(folder))

    if board == 'auto':
        board = which_board(folder)

    all_rhx_pd = fu.get_rhd_pd(folder, file_extension=board)

    logger.info('Found {} .{} files split in {} recordings'.format(len(all_rhx_pd.index),
                                                                   board,
                                                                   len(np.unique(all_rhx_pd['rec']))))

    if single_rec:
        logger.info('Will lump all files into single rec {}'.format(rec))
        all_rhx_pd.loc[:, 'rec'] = rec

    logger.info('dest file: {}'.format(dest_file_path))
    
    # go through a temporary file
    tmp_sess_dir = tempfile.mkdtemp()
    kwd_temp_path = os.path.join(tmp_sess_dir, os.path.split(dest_file_path)[-1])
    logger.info('tmp path {}'.format(kwd_temp_path))
    try:
        with h5py.File(kwd_temp_path, 'a') as kwd_temp_file: 
            first_header = intan_to_kwd_multirec(kwd_temp_file, all_rhx_pd,
                                                include_channels=include_channels)

        logger.info('moving back to {}'.format(dest_file_path))
        os.makedirs(os.path.split(dest_file_path)[0], exist_ok=True)
        shutil.move(kwd_temp_path, dest_file_path)
    finally:
        logger.info('removing temp file')
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(tmp_sess_dir)
    return first_header, all_rhx_pd
