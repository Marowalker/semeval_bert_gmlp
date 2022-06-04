TRAIN_SIZE = 8000
TEST_SIZE = 2717
MAX_SEN_LEN = 128
BATCH_SIZE = 16
DROPOUT = 0.1
LEARNING_RATE = 2e-5
NUM_EPOCH = 5
EMB_SIZE = 768
REBUILD = 0  # 1 to rebuild, 0 to load
REMAKE = 0  # 1 to build model again, 0 otherwise
TESTED = 1  # 1 if perl script has been run, 0 otherwise

relation = ['Other',
            'Message-Topic(e1,e2)',
            'Message-Topic(e2,e1)',
            'Product-Producer(e1,e2)',
            'Product-Producer(e2,e1)',
            'Instrument-Agency(e1,e2)',
            'Instrument-Agency(e2,e1)',
            'Entity-Destination(e1,e2)',
            'Entity-Destination(e2,e1)',
            'Cause-Effect(e1,e2)',
            'Cause-Effect(e2,e1)',
            'Component-Whole(e1,e2)',
            'Component-Whole(e2,e1)',
            'Entity-Origin(e1,e2)',
            'Entity-Origin(e2,e1)',
            'Member-Collection(e1,e2)',
            'Member-Collection(e2,e1)',
            'Content-Container(e1,e2)',
            'Content-Container(e2,e1)']

DATA = 'SemEval2010_task8_all_data/'
TRAIN_DATA = DATA + 'SemEval2010_task8_training/'
TEST_DATA = DATA + 'SemEval2010_task8_testing_keys/'
SCORE = DATA + 'SemEval2010_task8_scorer-v1.2/'
PROCESSED = 'data_processed/'
RAW_PROCESSED = PROCESSED + 'raw/'
PICKLE_PROCESSED = PROCESSED + 'pickle/'
TRAINED_MODELS = PROCESSED + 'trained_models/'

train_path = TRAIN_DATA + 'TRAIN_FILE.TXT'
train_x_ref_path = RAW_PROCESSED + 'train_x_ref.txt'
train_x_path = RAW_PROCESSED + 'train_x.txt'
train_pickle_x = PICKLE_PROCESSED + 'train_x_tok_ids'
train_pickle_x_pos = PICKLE_PROCESSED + 'train_x_pos'
train_y_path = RAW_PROCESSED + 'train_y.txt'
train_x_head_path = PICKLE_PROCESSED + 'train_x_head_mask'
train_x_e1_path = PICKLE_PROCESSED + 'train_x_e1_mask'
train_x_e2_path = PICKLE_PROCESSED + 'train_x_e2_mask'

test_path = TEST_DATA + 'TEST_FILE_FULL.TXT'
test_x_ref_path = RAW_PROCESSED + 'test_x_ref.txt'
test_x_path = RAW_PROCESSED + 'test_x.txt'
test_pickle_x = PICKLE_PROCESSED + 'test_x_tok_ids'
test_pickle_x_pos = PICKLE_PROCESSED + 'test_x_pos'
test_y_path = RAW_PROCESSED + 'test_y.txt'
test_x_head_path = PICKLE_PROCESSED + 'test_x_head_mask'
test_x_e1_path = PICKLE_PROCESSED + 'test_x_e1_mask'
test_x_e2_path = PICKLE_PROCESSED + 'test_x_e2_mask'
