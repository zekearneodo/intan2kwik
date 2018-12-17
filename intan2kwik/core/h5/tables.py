import logging
import h5py
import numpy as np

logger = logging.getLogger('intan2kwik.core.h5.tables')


def insert_table(group: h5py.Group, table: np.array, name: str, attr_dict: dict=None,
                 dtype: np.dtype=None) -> h5py.Dataset:
    dtype = table.dtype if dtype is None else dtype
    dset = group.create_dataset(name, data=table, dtype=dtype)
    if attr_dict is not None:
        append_atrributes(dset, attr_dict)
    return dset

def append_atrributes(h5obj, attr_dict_list):
    for attr_dict in attr_dict_list:
        # print attr_dict['name'] + ' {0} - {1}'.format(attr_dict['data'], attr_dict['dtype'])
        h5obj.attrs.create(attr_dict['name'], attr_dict['data'], dtype=attr_dict['dtype'])
        # h5obj.attrs.create(attr['name'], attr['data'], dtype=attr['dtype'])

def unlimited_rows_data(group, table_name, data):
    """
    Create a table with no max shape, to append data forever
    :param group: h5py Group object. parent group
    :param table_name: str. name of the table
    :param data: np.array with initial data. Can be empty
    :return:
    """
    logger.debug('Creating unbounded table {0} in group {1}'.format(
        group.name, table_name))
    try:
        table = group.create_dataset(table_name,
                                     shape=data.shape,
                                     dtype=data.dtype,
                                     maxshape=(None, None))
        table[:] = data

    except RuntimeError as e:
        if 'Name already exists' in str(e):
            logger.debug(
                'Table {} already exists, appending the data'.format(table_name))
            table = group[table_name]
            append_rows(table, data)
        else:
            raise
    return table


def append_rows(dataset, new_data):
    '''
    Append rows to an existing table
    :param dataset: h5py Dataset object. Where to append the data
    :param new_data: array. An array to append
    :return:
    '''
    rows = dataset.shape[0]
    more_rows = new_data.shape[0]
    logger.debug('Appending {} rows to dataset {}'.format(
        more_rows, dataset.name))
    dataset.resize(rows + more_rows, axis=0)
    if dataset.size == (rows + more_rows):
        dataset[rows:] = new_data
    else:
        dataset[rows:, :] = new_data
