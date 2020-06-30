

def data_process():
    with open('./train_data_', 'w') as writer:
        with open('./train_data', 'r') as reader:
            for line in reader.readlines():
                split_arr = line.split(' ')
                if len(split_arr) != 4:
                    # print('\n')
                    writer.write('\n')
                else:
                    # print(split_arr[0], split_arr[1], split_arr[2], split_arr[3].replace('-', '_'))
                    writer.write(split_arr[0] + ' ' + split_arr[1] + ' ' + split_arr[2] + ' ' + split_arr[3].replace('-', '_'))

data_process()