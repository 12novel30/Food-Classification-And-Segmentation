# food classification paramaters
param_classification = {
    'food_root':'/home/haeun/21winter/data_from/food_classification/',
    'batch_size':10,
    'food_classes':51,
    'num_epochs':25,
    'model_folder':'/home/haeun/21winter/week7/c_models/',
    'weight_':'weights.pt',
    'graph_path':'/home/haeun/21winter/week7/c_inferences/'
}

# food segmentation parameters
param_segmentation = {
    # model
    'model_1,2_pre_path':'/home/haeun/21winter/week7/s_models/1_2/Unseen_Food_Segmentation_MASKRCNN_synthetic_data.tar',
    'model_1,2_dir':'/home/haeun/21winter/week7/s_models/1_2/',
    'model_3,4,5_dir':'/home/haeun/21winter/week7/s_models/3_4_5/',
    'model_6,7_dir':'/home/haeun/21winter/week7/s_models/6_7/',
    # best model은 파일명 앞에 best_ 붙이는 방식으로 수정하기
    
    # real data
    'real_food_root':'/home/haeun/21winter/data_from/food_segmentation_real/food_segmentation_real/',
    'cycle1':'seg_food_cycle1',
    'cycle2':'seg_food_cycle2',
    'addition':'seg_food_addition',
    'image_':'RGBImages',
    'mask_':'Annotations/Annotations_all',
    # syn data
    'num_test':all,
    
    # inference data
    '2_inf_dir':'/home/haeun/21winter/data_from/food_segmentation_real/food_segmentation_real/seg_food_cycle1/RGBImages',
    '3,4,5_inf_dir':'/home/haeun/21winter/data_from/food_segmentation_real/food_segmentation_real/seg_food_cycle2/RGBImages',
    '6,7_inf_dir':'/home/haeun/21winter/data_from/food_segmentation_synthetic/image',
    
    
    
    
    'width':640,
    'height':480,
    'num_class':2,
    'start_epoch':0,
    'max_epoch':40,
    'save_interval':1,
    'save_dir':'/home/haeun/21winter/week7/s_models/0215_13:30',
    # for inference
    '0215_13:30_best_model':'/home/haeun/21winter/week7/s_models/0215_13:30/epoch_35.tar',
    # seg_real_cycle1
    'inf_path1':'/home/haeun/21winter/data_from/food_segmentation_real/food_segmentation_real/seg_food_cycle1/RGBImages',
    'output_dir':'/home/haeun/21winter/week7/s_models/',

    'output_root':'/home/haeun/21winter/week7/s_models/',
    'c_model_path1':'/home/haeun/21winter/week6/classification/models/weights.pt',
    'c_train_data_path':'/home/haeun/21winter/data_from/food_classification/',
    'food_classes':51,
}

# food index
food_id = {
            '0':'오징어/낙지',
            '1':'흰밥',
            '2':'잡곡밥',
            '3':'죽',
            '4':'미역국',
            '5':'된장국',
            '6':'스프',
            '7':'북엇국',
            '8':'감자양파국',
            '9':'떡국',
            '10':'고추장찌개',
            '11':'배추김치',
            '12':'무김치',
            '13':'열무김치',
            '14':'오이김치',
            '15':'물김치',
            '16':'샐러드',
            '17':'김',
            '18':'숙주나물',
            '19':'가지무침',
            '20':'마늘쫑무침',
            '21':'버섯볶음',
            '22':'브로콜리',
            '23':'호박볶음',
            '24':'무생채',
            '25':'감자채볶음',
            '26':'으깸샐러드',
            '27':'부추생채',
            '28':'양파절임',
            '29':'웻지감자',
            '30':'과일',
            '31':'두부',
            '32':'콩조림',
            '33':'스크램블에그',
            '34':'계란후라이',
            '35':'계란찜',
            '36':'삶은계란',
            '37':'계란말이',
            '38':'떡갈비',
            '39':'돈까스',
            '40':'제육볶음',
            '41':'장조림',
            '42':'탕수육',
            '43':'닭볶음',
            '44':'소시지볶음',
            '45':'돼지갈비찜',
            '46':'진미채무침',
            '47':'동태전',
            '48':'멸치볶음',
            '49':'어묵볶음',
            '50':'볶음밥'
}

