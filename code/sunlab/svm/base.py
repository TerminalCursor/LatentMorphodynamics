def svm_load(scaler_path, clf_path):
    """# Load SVM Models
    Load the scaler and CLF
    """
    from onnxruntime import InferenceSession
    from numpy import float32
    with open(scaler_path, "rb") as f:
        onx_scaler = f.read()
    with open(clf_path, "rb") as f:
        onx_clf = f.read()
    sess_scaler = InferenceSession(onx_scaler, providers=["CPUExecutionProvider"])
    sess_clf = InferenceSession(onx_clf, providers=["CPUExecutionProvider"])
    scaler_predictor = lambda X: sess_scaler.run(None, {"X": X.astype(float32)})
    clf_predictor    = lambda X:    sess_clf.run(None, {"X": X.astype(float32)})
    return scaler_predictor, clf_predictor

def svm_infer(data, d=1, pixel2dist=1.076268, scaler_model=None, clf_model=None):
    """# Infer from SVM Models)"""
    from sunlab.transform_data import Add_Measures, Exclude_Measures
    from numpy import array
    #scale columns by following
    data[:,0] =data[:,0]*pixel2dist*pixel2dist#area
    data[:,1] = data[:,1]*pixel2dist#majoraxislength
    data[:,2] = data[:,2]*pixel2dist#minoraxislength
    #3 is eccentricity, unitless
    data[:,4] =data[:,4]*pixel2dist*pixel2dist#4 convex area,
    data[:,5] = data[:,5]*pixel2dist#5 Equiv diam
    #6 solidity, unitless
    #7 extent, unitless
    data[:,8] = data[:,8]*pixel2dist#8 Perimeter
    data[:,9] = data[:,9]*pixel2dist#9 Convex Perimeter
    data[:,10] = data[:,10]*pixel2dist#10 Fiber Length
    data[:,11] = data[:,11]*pixel2dist#11 Max Inscribed Radius
    data[:,12] = data[:,12]*pixel2dist#12 Bleb_length


    #add form factor as last column of data?
    #Form factor is added inside the Processing data when doing SVM. See "Transform_data.py"

    #if len(Measure_Delete)>0:
    #	data = np.delete(data, Measure_Delete, axis = 1) #delete time column and cellnumber column, since we already have them.

    #so now data should look just like other data used in SVM

    #add aspect ratio as last column of data
    X_data = Add_Measures(data, add_AR=True, add_FF=True, add_convexity=True,
                            add_curl_old=True, add_curl=True, add_sphericity=True,
                            add_InscribedArea=True, add_BlebRel=True)
    #if you wish to exclude certain measures:
    #Area,MjrAxis,MnrAxis,Ecc,ConA,EqD,Sol,Ext,Per,conPer,FL,InR,bleb_M
    X_data = Exclude_Measures(X_data, ex_Area=False,
                            ex_MjrAxis=False, ex_MnrAxis=False, ex_Ecc=False,
                            ex_ConA=False, ex_EqD=False, ex_Sol=False, ex_Ext=False,
                            ex_Per=False,ex_conPer=False,ex_FL=False,ex_InR=False,
                            ex_bleb=False)

    ####IF THE DATA WAS POLYNOMIAL BEFORE SCALED, DO THAT NOW!
    #frameinfo = getframeinfo(currentframe())
    #print("IF YOUR SCALER IS A POLYNOMIAL, YOU NEED TO EDIT THE POLYNOMIAL FEATURES, LINE %d CODE" % (frameinfo.lineno + 2))
    #d = 1
    if d==2:
        print("Expanding feature set to include quadratic, cross terms.")
        poly=preprocessing.PolynomialFeatures(degree = d, interaction_only = True)
        X_data_exp = poly.fit_transform(X_data)

        #FIRST, SCALE THE DATA USING THE SCALER
        X_data_scaled = scaler_model(X_data_exp)[0]
    else:
        X_data_scaled = scaler_model(X_data)[0]

    #GATHER PROBABILITIES
    Probs = clf_model(X_data_scaled)[1]
            
#     print("Probs")
#     print(Probs)
    Probs = array([list(v.values()) for v in Probs])

    #Gather Predictions
    Predictions = clf_model(X_data_scaled)[0]

    Descriptors = ['frame', 'cellnumber','x-cent','y-cent','actinedge','filopodia','bleb','lamellipodia']
    return Probs