
param = {
    'train_way': 5,
    'train_support': 20,
    'train_query': 2,

    'test_way': 5,
    'test_support': 20,
    'test_query':20,

    # 'test_labels': ['Empty', 'Walking', 'Sitting', 'Lying', 'Standing'],
    'test_labels': ['Empty', 'Lying', 'Sitting', 'Standing', 'Walking'],

    # #ReWis setting
    # 'test_way': 4,
    # 'test_support': 8,
    # 'test_query': 8,
    # 'test_labels': ['empty', 'jump', 'stand', 'walk'],

    # model
    'max_epoch' : 10,
    'epoch_size' : 1000,
    'lr': 0.0001
}
