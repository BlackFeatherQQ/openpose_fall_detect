class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels ~~~video数据源
            root_dir = 'D:/agnewwwwwwwwwwwww/data/video'

            # Save preprocess data into output_dir   ~~~图片保存源
            output_dir = 'D:/agnewwwwwwwwwwwww/data/graph'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/Path/to/hmdb-51'

            output_dir = '/path/to/VAR/hmdb51'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './Models/c3d-pretrained.pth'

#参考用