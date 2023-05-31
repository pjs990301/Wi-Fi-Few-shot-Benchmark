
param = {
    'train_way': 4,
    'train_support': 8,
    'train_query': 8,

    # 'test_way': 5,
    # 'test_support': 10,
    # 'test_query':10,

    # ReWiS 정상
    'test_way': 4,
    'test_support': 4,
    'test_query':4,

    # 'test_labels': ['Empty', 'Lying', 'Sitting', 'Standing', 'Walking'],
    'test_labels': ['empty', 'jump', 'stand', 'walk'],

    # model
    'max_epoch' : 7,
    'epoch_size' : 500,
    'lr': 0.0001,
    # 'max_epoch' : 1,
    # 'epoch_size' : 10,
}
