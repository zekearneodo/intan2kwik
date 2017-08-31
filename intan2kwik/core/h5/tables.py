import logging
import h5py

logger = logging.getLogger('intan2kwik.core.h5.tables')


def unlimited_rows_data(group, table_name, data):
    """
    Create a table with no max shape, to append data forever
    :param group: h5py Group object. parent group
    :param table_name: str. name of the table
    :param data: np.array with initial data. Can be empty
    :return:
    """
    logger.debug('Creating unbounded table {0} in group {1}'.format(group.name, table_name))
    try:
        table = group.create_dataset(table_name,
                                     shape=data.shape,
                                     dtype=data.dtype,
                                     maxshape=(None, None))
        table[:] = data

    except RuntimeError as e:
        if 'Name already exists' in str(e):
            logger.debug('Table {} already exists, appending the data'.format(table_name))
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
    logger.debug('Appending {} rows to dataset {}'.format(more_rows, dataset.name))
    dataset.resize(rows + more_rows, axis=0)
    if dataset.size == (rows + more_rows):
        dataset[rows:] = new_data
    else:
        dataset[rows:, :] = new_data
