DIR_ROOT = "../"
FILES = {
    "SAMPLE_DATA": {
        "3mgmL_subsampled": DIR_ROOT + "data/sample_data/Spheroid_3p0mgmL_2kr.csv"
    },
    "TRAINING_DATASET": DIR_ROOT + "data/spheroid26_011523_filtered.csv",
    "TRAINING_DATASET_WIDE_BERTH": DIR_ROOT + "data/spheroid26_011523_exc.csv",
    "PRETRAINED_MODEL_DIR": DIR_ROOT + "models/current_model/",
    "PEN_TRACKED": {
        "AUTO": DIR_ROOT + "data/PEN_automatically_tracked.csv",
        "MANUAL": DIR_ROOT + "data/PEN_manually_tracked.csv",
    },
    "RHO": {
        "3": DIR_ROOT + "data/Rho_act_Cell3.csv",
        "4": DIR_ROOT + "data/Rho_act_Cell4.csv",
        "6": DIR_ROOT + "data/Rho_act_Cell6.csv",
        "7": DIR_ROOT + "data/Rho_act_Cell7.csv",
        "8": DIR_ROOT + "data/Rho_act_Cell8.csv",
        "9": DIR_ROOT + "data/Rho_act_Cell9.csv",
        "10": DIR_ROOT + "data/Rho_act_Cell10.csv",
        "11": DIR_ROOT + "data/Rho_act_Cell11.csv",
    },
    "CN03": {
        "3": DIR_ROOT + "data/Rho_act_Cell3.csv",
        "4": DIR_ROOT + "data/Rho_act_Cell4.csv",
        "6": DIR_ROOT + "data/Rho_act_Cell6.csv",
        "7": DIR_ROOT + "data/Rho_act_Cell7.csv",
        "8": DIR_ROOT + "data/Rho_act_Cell8.csv",
        "9": DIR_ROOT + "data/Rho_act_Cell9.csv",
        "10": DIR_ROOT + "data/Rho_act_Cell10.csv",
        "11": DIR_ROOT + "data/Rho_act_Cell11.csv",
    },
    "Y27632": {
        "1": DIR_ROOT + "data/Y27632_Cell1.csv",
        "2": DIR_ROOT + "data/Y27632_Cell2.csv",
        "3": DIR_ROOT + "data/Y27632_Cell3.csv",
        "4": DIR_ROOT + "data/Y27632_Cell4.csv",
        "5": DIR_ROOT + "data/Y27632_Cell5.csv",
    },
    "HISTOLOGIES": {
        "J1": DIR_ROOT + "../HistologyProject/J1/svm/svm.csv",
        "image001": DIR_ROOT + "../HistologyProject/image001/svm/svm.csv",
        "H4": DIR_ROOT + "../HistologyProject/H4/svm/svm.csv",
        "H9": DIR_ROOT + "../HistologyProject/H9/svm/svm.csv",
    },
    "SPHEROID": {
        "1p5mgml": DIR_ROOT + "data/spheroid26_011523_filtered.csv",
        "3mgml": DIR_ROOT + "data/spheroid20_011923_filtered.csv",
        "4mgml": DIR_ROOT + "data/spheroid22_012323_filtered.csv",
        "6mgml": DIR_ROOT + "data/spheroid26_012423_filtered.csv",
    },
    "SPHEROID_RAW": {
        "REG": {
            "1p5mgml": [
                DIR_ROOT + "data/" + "spheroid15_030322_RegularCollagen_1p5mgml.csv",
                DIR_ROOT + "data/" + "spheroid16_030322_RegularCollagen_1p5mgml.csv",
                DIR_ROOT + "data/" + "spheroid17_041022_RegularCollagen_1p5mgml.csv",
                DIR_ROOT + "data/" + "spheroid18_041022_RegularCollagen_1p5mgml.csv",
                DIR_ROOT + "data/" + "spheroid26_011523_RegularCollagen_1p5mgml.csv",
                DIR_ROOT + "data/" + "spheroid27_090323_RegularCollagen_1p5mgml.csv",
                DIR_ROOT + "data/" + "spheroid28_090523_RegularCollagen_1p5mgml.csv",
            ],
            "3mgml": [
                DIR_ROOT + "data/" + "spheroid15_031022_RegularCollagen_3mgml.csv",
                DIR_ROOT + "data/" + "spheroid16_031522_RegularCollagen_3mgml.csv",
                DIR_ROOT + "data/" + "spheroid17_041022_RegularCollagen_3mgml.csv",
                DIR_ROOT + "data/" + "spheroid18_083022_RegularCollagen_3mgml.csv",
                DIR_ROOT + "data/" + "spheroid19_083022_RegularCollagen_3mgml.csv",
                DIR_ROOT + "data/" + "spheroid20_011923_RegularCollagen_3mgml.csv",
            ],
            "4mgml": [
                DIR_ROOT + "data/" + "spheroid17_031022_RegularCollagen_4mgml.csv",
                DIR_ROOT + "data/" + "spheroid18_031022_RegularCollagen_4mgml.csv",
                DIR_ROOT + "data/" + "spheroid19_031022_RegularCollagen_4mgml.csv",
                DIR_ROOT + "data/" + "spheroid20_083022_RegularCollagen_4mgml.csv",
                DIR_ROOT + "data/" + "spheroid21_083022_RegularCollagen_4mgml.csv",
                DIR_ROOT + "data/" + "spheroid22_012323_RegularCollagen_4mgml.csv",
            ],
            "6mgml": [
                DIR_ROOT + "data/" + "spheroid22_031022_RegularCollagen_6mgml.csv",
                DIR_ROOT + "data/" + "spheroid23_031022_RegularCollagen_6mgml.csv",
                DIR_ROOT + "data/" + "spheroid24_031022_RegularCollagen_6mgml.csv",
                DIR_ROOT + "data/" + "spheroid25_031022_RegularCollagen_6mgml.csv",
                DIR_ROOT + "data/" + "spheroid26_012423_RegularCollagen_6mgml.csv",
            ],
        },
        "PC": {
            "1p5mgml": [
                DIR_ROOT + "data/" + "spheroid1_021922_PhotoCol_1p5mgml.csv",
                DIR_ROOT + "data/" + "spheroid2_021922_PhotoCol_1p5mgml.csv",
                DIR_ROOT + "data/" + "spheroid3_021922_PhotoCol_1p5mgml.csv",
                DIR_ROOT + "data/" + "spheroid4_021922_PhotoCol_1p5mgml.csv",
                DIR_ROOT + "data/" + "spheroid5_021922_PhotoCol_1p5mgml.csv",
                DIR_ROOT + "data/" + "spheroid7_021922_PhotoCol_1p5mgml.csv",
                DIR_ROOT + "data/" + "spheroid8_021922_PhotoCol_1p5mgml.csv",
                DIR_ROOT + "data/" + "spheroid9_012723_PhotoCol_1p5mgml.csv",
            ],
            "3mgml": [
                DIR_ROOT + "data/" + "spheroid5_022022_PhotoCol_3mgml.csv",
                DIR_ROOT + "data/" + "spheroid6_022022_PhotoCol_3mgml.csv",
                DIR_ROOT + "data/" + "spheroid7_030322_PhotoCol_3mgml.csv",
                DIR_ROOT + "data/" + "spheroid8_030322_PhotoCol_3mgml.csv",
                DIR_ROOT + "data/" + "spheroid12_060923_PhotoCol_3mgml.csv",
            ],
            "4mgml": [
                DIR_ROOT + "data/" + "spheroid2_022022_PhotoCol_4mgml.csv",
                DIR_ROOT + "data/" + "spheroid3_053122_PhotoCol_4mgml.csv",
                DIR_ROOT + "data/" + "spheroid4_053122_PhotoCol_4mgml.csv",
                DIR_ROOT + "data/" + "spheroid5_053122_PhotoCol_4mgml.csv",
                DIR_ROOT + "data/" + "spheroid6_053122_PhotoCol_4mgml.csv",
                DIR_ROOT + "data/" + "spheroid7_091222_PhotoCol_4mgml.csv",
                DIR_ROOT + "data/" + "spheroid8_091822_PhotoCol_4mgml.csv",
            ],
            "6mgml": [
                DIR_ROOT + "data/" + "spheroid1_021922_PhotoCol_6mgml.csv",
                DIR_ROOT + "data/" + "spheroid2_021922_PhotoCol_6mgml.csv",
                DIR_ROOT + "data/" + "spheroid3_021922_PhotoCol_6mgml.csv",
                DIR_ROOT + "data/" + "spheroid4_021922_PhotoCol_6mgml.csv",
                DIR_ROOT + "data/" + "spheroid6_091222_PhotoCol_6mgml.csv",
                DIR_ROOT + "data/" + "spheroid8_022323_PhotoCol_6mgml.csv",
            ],
        },
    },
    "SPHEROID_RAW_RIC": {
        "NAME": "(R)esolution (I)mage type (C)oncentration",
        "QUARTER_HOUR": {
            "REG": {
                "1p5mgml": [
                    DIR_ROOT
                    + "data/"
                    + "spheroid28_090523_RegularCollagen_1p5mgml.csv",
                ],
                "3mgml": [
                    DIR_ROOT + "data/" + "spheroid20_011923_RegularCollagen_3mgml.csv",
                ],
                "4mgml": [
                    DIR_ROOT + "data/" + "spheroid22_012323_RegularCollagen_4mgml.csv",
                ],
                "6mgml": [
                    DIR_ROOT + "data/" + "spheroid26_012423_RegularCollagen_6mgml.csv",
                ],
            },
            "PC": {
                "1p5mgml": [
                    DIR_ROOT + "data/" + "spheroid9_012723_PhotoCol_1p5mgml.csv",
                ],
                "3mgml": [
                    DIR_ROOT + "data/" + "spheroid12_060923_PhotoCol_3mgml.csv",
                ],
                "4mgml": [],
                "6mgml": [
                    DIR_ROOT + "data/" + "spheroid8_022323_PhotoCol_6mgml.csv",
                ],
            },
        },
        "DAILY": {
            "REG": {
                "1p5mgml": [
                    DIR_ROOT
                    + "data/"
                    + "spheroid15_030322_RegularCollagen_1p5mgml.csv",
                    DIR_ROOT
                    + "data/"
                    + "spheroid16_030322_RegularCollagen_1p5mgml.csv",
                    DIR_ROOT
                    + "data/"
                    + "spheroid17_041022_RegularCollagen_1p5mgml.csv",
                    DIR_ROOT
                    + "data/"
                    + "spheroid18_041022_RegularCollagen_1p5mgml.csv",
                    DIR_ROOT
                    + "data/"
                    + "spheroid26_011523_RegularCollagen_1p5mgml.csv",
                    DIR_ROOT
                    + "data/"
                    + "spheroid27_090323_RegularCollagen_1p5mgml.csv",
                ],
                "3mgml": [
                    DIR_ROOT + "data/" + "spheroid15_031022_RegularCollagen_3mgml.csv",
                    DIR_ROOT + "data/" + "spheroid16_031522_RegularCollagen_3mgml.csv",
                    DIR_ROOT + "data/" + "spheroid17_041022_RegularCollagen_3mgml.csv",
                    DIR_ROOT + "data/" + "spheroid18_083022_RegularCollagen_3mgml.csv",
                    DIR_ROOT + "data/" + "spheroid19_083022_RegularCollagen_3mgml.csv",
                ],
                "4mgml": [
                    DIR_ROOT + "data/" + "spheroid17_031022_RegularCollagen_4mgml.csv",
                    DIR_ROOT + "data/" + "spheroid18_031022_RegularCollagen_4mgml.csv",
                    DIR_ROOT + "data/" + "spheroid19_031022_RegularCollagen_4mgml.csv",
                    DIR_ROOT + "data/" + "spheroid20_083022_RegularCollagen_4mgml.csv",
                    DIR_ROOT + "data/" + "spheroid21_083022_RegularCollagen_4mgml.csv",
                ],
                "6mgml": [
                    DIR_ROOT + "data/" + "spheroid22_031022_RegularCollagen_6mgml.csv",
                    DIR_ROOT + "data/" + "spheroid23_031022_RegularCollagen_6mgml.csv",
                    DIR_ROOT + "data/" + "spheroid24_031022_RegularCollagen_6mgml.csv",
                    DIR_ROOT + "data/" + "spheroid25_031022_RegularCollagen_6mgml.csv",
                ],
            },
            "PC": {
                "1p5mgml": [
                    DIR_ROOT + "data/" + "spheroid1_021922_PhotoCol_1p5mgml.csv",
                    DIR_ROOT + "data/" + "spheroid2_021922_PhotoCol_1p5mgml.csv",
                    DIR_ROOT + "data/" + "spheroid3_021922_PhotoCol_1p5mgml.csv",
                    DIR_ROOT + "data/" + "spheroid4_021922_PhotoCol_1p5mgml.csv",
                    DIR_ROOT + "data/" + "spheroid5_021922_PhotoCol_1p5mgml.csv",
                    DIR_ROOT + "data/" + "spheroid7_021922_PhotoCol_1p5mgml.csv",
                    DIR_ROOT + "data/" + "spheroid8_021922_PhotoCol_1p5mgml.csv",
                ],
                "3mgml": [
                    DIR_ROOT + "data/" + "spheroid5_022022_PhotoCol_3mgml.csv",
                    DIR_ROOT + "data/" + "spheroid6_022022_PhotoCol_3mgml.csv",
                    DIR_ROOT + "data/" + "spheroid7_030322_PhotoCol_3mgml.csv",
                    DIR_ROOT + "data/" + "spheroid8_030322_PhotoCol_3mgml.csv",
                    DIR_ROOT + "data/" + "spheroid12_060923_PhotoCol_3mgml.csv",
                ],
                "4mgml": [
                    DIR_ROOT + "data/" + "spheroid2_022022_PhotoCol_4mgml.csv",
                    DIR_ROOT + "data/" + "spheroid3_053122_PhotoCol_4mgml.csv",
                    DIR_ROOT + "data/" + "spheroid4_053122_PhotoCol_4mgml.csv",
                    DIR_ROOT + "data/" + "spheroid5_053122_PhotoCol_4mgml.csv",
                    DIR_ROOT + "data/" + "spheroid6_053122_PhotoCol_4mgml.csv",
                    DIR_ROOT + "data/" + "spheroid7_091222_PhotoCol_4mgml.csv",
                ],
                "6mgml": [
                    DIR_ROOT + "data/" + "spheroid1_021922_PhotoCol_6mgml.csv",
                    DIR_ROOT + "data/" + "spheroid2_021922_PhotoCol_6mgml.csv",
                    DIR_ROOT + "data/" + "spheroid3_021922_PhotoCol_6mgml.csv",
                    DIR_ROOT + "data/" + "spheroid4_021922_PhotoCol_6mgml.csv",
                    DIR_ROOT + "data/" + "spheroid6_091222_PhotoCol_6mgml.csv",
                ],
            },
        },
    },
    "SPHEROID_PC": {
        "1p5mgml": DIR_ROOT + "data/spheroid9_012723_pc_1p5.csv",
        "3mgml": DIR_ROOT + "data/spheroid12_060923_pc_3.csv",
    },
    "SPHEROID_MASKS": {
        "1p5mgml": DIR_ROOT + "data/spheroid_1p5mgml_spheroid26_011523/images/",
        "3mgml": DIR_ROOT + "data/spheroid_3mgml_spheroid20_011923/images/",
        "4mgml": DIR_ROOT + "data/spheroid_4mgml_spheroid22_012323/images/",
        "6mgml": DIR_ROOT + "data/spheroid_6mgml_spheroid26_012423/images/",
        "3mgml_pc": DIR_ROOT + "data/spheroid_photocol_3mgml_spheroid12_060923/images/",
    },
    "FIGURES": {
        "2": {
            "PHENOTYPES_SMOOTHED": DIR_ROOT
            + "extra_data/PhenotypeGaussianSmoothed.npy",
        },
        "3": {
            "SAMPLED_DATASET": DIR_ROOT + "extra_data/Figure3.SampledDataset.npy",
            "PAIRWISE_DISTANCES": DIR_ROOT + "extra_data/Figure3.PairwiseDistances.npy",
            "PAIRWISE_DOT_PRODUCTS": DIR_ROOT
            + "extra_data/Figure3.PairwiseDotProducts.npy",
            "TRAJECTORIES": DIR_ROOT + "extra_data/Figure3.Trajectories.npy",
        },
    },
    "PHENOTYPE_GRID": {
        "IN": DIR_ROOT + "extra_data/PhenotypeGrid.npy",
        "OUT": DIR_ROOT + "extra_data/PhenotypeGrid_out.npy",
    },
    "PHENOTYPE_RGB": DIR_ROOT + "extra_data/PhenotypeColors.npy",
    "SVM": {
        "MODEL": DIR_ROOT + "other/svm/SVC_rbf_010820_16942_new.pkl",
        "SCALER": DIR_ROOT + "other/svm/SVC_rbf_scaler_010820_16942_new.pkl",
        "ONNX": {
            "MODEL": DIR_ROOT + "models/svm/SVM_clf.onnx",
            "SCALER": DIR_ROOT + "models/svm/SVM_scaler.onnx",
        },
    },
    "NONPHYSICAL_MASK": DIR_ROOT + "extra_data/NonPhysicalMask.npy",
}
