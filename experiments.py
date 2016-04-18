import sys
import os

from data import DataList

NFEATURES = 10000

class Experiments:

    def __init__(self, split_by_class = False, data_format = 'sgm'):

        self.train_data = []
        self.heldout_data = [] # it can be a pool as well
        self.test_data = []
        self.labels = []
        self.split = split_by_class
        self.data_format = data_format
        self.data_obj = DataList(NFEATURES)



    def read_data(self, dir_name, out_file = 'model_svm.lsv'):



        # read the files with text data
        for subdir, dirs, files in os.walk(dir_name):
            for f in files:
                file_path = subdir + os.path.sep + f
                print(file_path)
                self.data_obj.read(file_path)

        self.data_obj.vectorize()
        self.write_data(out_file)

    def write_data(self, out_file):
        """
        Write features in a libsvm format
        """

        self.data_obj.write_data(out_file)

    def load_data(self, out_file):
        """
        Write features in a libsvm format
        """

        self.data_obj.load_data(out_file)


    def perform(self, dir_name = '', data_file = '', class_names  = []):
 
        
        if data_file == '' and dir_name != '':
            self.read_data(dir_name)
        elif dir_name == '' and data_file != '':
            self.load_data(data_file)
        else:
            print('Something is wrong with your data entry...')
            return -1

        for c in class_names:
            train_p, train_n, test_p, test_n = self.data_obj.split_by_class(c)
            # it returns a model and a new train data
            # TODO : shuffle train data ?
            svm_model, train_p, train_n = self.active_train(train_p, train_n)
            
    
if __name__ == '__main__':
    option = sys.argv[1]

    e = Experiments()
    if option == '-d':
        dir_name = sys.argv[2]
        e.perform(dir_name = dir_name)
    elif option == '-f':
        data_file = sys.argv[2]
        e.perform(data_file = dir_name)
    else:
        print('Something is worng with your option...')
