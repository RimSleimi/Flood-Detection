import streamlit as st
import matplotlib.pyplot as plt
import rasterio
import onnxruntime
import time
import numpy as np



def app():
    st.title("Flood Detection")
    
    st.subheader("Upload Sentinel 1 data")
      
    col1, col2 = st.columns(2)
    uploaded_VH = col1.file_uploader("Upload Sentinel 1 VH band", type=["tif"])   
    uploaded_VV = col2.file_uploader("Upload Sentinel 1 VV band", type=["tif"])
    
    start_time = time.time()
    
      
    def process_bands(raw_VH, raw_VV):
        
        VV = (np.clip(raw_VV, -50, 1) + 50) / 51
        VH = (np.clip(raw_VH, -50, 1) + 50) / 51

        VV = (VV - 0.6851) / 0.0820
        VH = (VH - 0.6851) / 0.1102
        RFCC = np.stack((VV, VH, VH-VV), axis=2)   # (512, 512, 3)

        return RFCC
    
    
    ort_session = onnxruntime.InferenceSession('MANet_resnet50.onnx')
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    @st.cache(allow_output_mutation=True)
    def predict(input_data):
        return sess.run([output_name], {input_name: input_data})[0]

   
    if uploaded_VH and uploaded_VV:
        raw_VH = np.nan_to_num(rasterio.open(uploaded_VH).read())[0]
        raw_VV = np.nan_to_num(rasterio.open(uploaded_VV).read())[0]

        RFCC = process_bands(raw_VH, raw_VV)

        fig, ax = plt.subplots(1,3, figsize = (20,10)) 

        ax[0].imshow(raw_VV, cmap='gray') 
        ax[0].set_title('Raw VV')
        ax[0].axis('off')

        ax[1].imshow(raw_VH, cmap='gray') 
        ax[1].set_title('Raw VH')
        ax[1].axis('off')

        ax[2].imshow(RFCC) 
        ax[2].set_title('RFCC')
        ax[2].axis('off')

        st.pyplot(fig)
        
        
        get_mask = st.button('Predict', key='Inference')
 
        if get_mask: 
            with st.spinner('Wait for it...'):
                ort_inputs = {'modelInput': RFCC.transpose((2, 0, 1))[None].astype(np.float32)}
                ort_outs = ort_session.run(None, ort_inputs)
                preds = ort_outs[0]

                fig, ax = plt.subplots(1,2, figsize = (10,10)) 

                ax[0].imshow(RFCC) 
                ax[0].set_title('RFCC')
                ax[0].axis('off')

                ax[1].imshow(preds[0][0], cmap='gray') 
                ax[1].set_title('Prediction')
                ax[1].axis('off')
                
                st.pyplot(fig)

   
    end_time = time.time()
    execution_time = end_time - start_time
    st.success('Execution Time: ' + str("%.2f" % execution_time) + ' seconds', icon=None) 
