import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split



def train_val_files(args):
    
    c2smsfloods_source = sorted(glob(args.base_dir1))[:900]
#     c2smsfloods_source = [os.path.join(path, '*/') for path in c2smsfloods]

    c2smsfloods_labels = [path.replace('c2smsfloods_v1_source_s1',
                                       'c2smsfloods_v1_labels_s1_water') for path in c2smsfloods_source]
    
    S1Hand_labeled = os.path.join(args.base_dir2, 'HandLabeled/S1Hand/*')
    Hand_labeles = os.path.join(args.base_dir2, 'HandLabeled/LabelHand/*')

    S1Weakly_labeled = os.path.join(args.base_dir2, 'WeaklyLabeled/S1Weak/*')
    Weakly_labeles = os.path.join(args.base_dir2, 'WeaklyLabeled/S1OtsuLabelWeak/*')
    
    train_source = sorted(glob(S1Hand_labeled))[15:] + sorted(glob(S1Weakly_labeled)) + c2smsfloods_source
    train_labels = sorted(glob(Hand_labeles))[15:] + sorted(glob(Weakly_labeles)) + c2smsfloods_labels

    test_source = sorted(glob(S1Hand_labeled))[:16]
    test_labels = sorted(glob(Hand_labeles))[:16]
    
    test_df = pd.DataFrame({'source': test_source, 'label':test_labels})
    
    df = pd.DataFrame({'source': train_source, 'label':train_labels})
    train_df, val_df = train_test_split(df,
                                        test_size=0.1,
                                        random_state=args.seed,
                                        shuffle=True)

    return train_df, val_df, test_df