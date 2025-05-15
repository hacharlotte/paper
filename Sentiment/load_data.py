from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
import random
from torch.utils.data import DataLoader
from itertools import chain
class ABSA_DataLoader(DataLoader):
    def __init__(self, dataset, sort_key, sort_bs_num=None, is_shuffle=True, **kwargs):
        '''
        :param dataset: Dataset object
        :param sort_idx: sort_function
        :param sort_bs_num: sort range; default is None(sort for all sequence)
        :param is_shuffle: shuffle chunk , default if True
        :return:
        '''
        assert isinstance(dataset.data, list)
        super().__init__(dataset,**kwargs)
        self.sort_key = sort_key
        self.sort_bs_num = sort_bs_num
        self.is_shuffle = is_shuffle

    def __iter__(self):
        if self.is_shuffle:
            self.dataset.data = self.block_shuffle(self.dataset.data, self.batch_size, self.sort_bs_num, self.sort_key, self.is_shuffle)

        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)

    @staticmethod
    def block_shuffle(data, batch_size, sort_bs_num, sort_key, is_shuffle):
        # sort
        random.shuffle(data)
        data = sorted(data, key = sort_key) # 先按照长度排序
        batch_data = [data[i : i + batch_size] for i in range(0,len(data),batch_size)]
        batch_data = [sorted(batch, key = sort_key) for batch in batch_data]
        if is_shuffle:
            random.shuffle(batch_data)
        batch_data = list(chain(*batch_data))
        return batch_data
def prepare_dataset(train,test,vaild,config, absa_dataset, collate_fn):
    train_path = train
    dev_path = vaild
    test_path = test

    train_loader = DataLoader(absa_dataset(train_path),
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=collate_fn) if train_path else None
    dev_loader = DataLoader(absa_dataset(dev_path),
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=collate_fn) if dev_path else None
    test_loader = DataLoader(absa_dataset(test_path),
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=collate_fn) if test_path else None
    if dev_loader is None and test_loader is not None:
        dev_loader = test_loader
    return train_loader, dev_loader, test_loader