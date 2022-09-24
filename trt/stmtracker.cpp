#include "stmtracker.hpp"
// #include "utils.hpp"


#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#define batch_size 1
stmtracker::stmtracker(/* args */)
{

    // saveEngineFile("../../backbone_q.onnx","../../backbone_q.engine");
    // saveEngineFile("../../memorize.onnx","../../memorize.engine");
    // saveEngineFile("../../head.onnx","../../head.engine");

    // parseOnnxModel(model_path_base_q,1U<<24,engine_base_q,context_base_q);
    parseOnnxModelINT8(model_path_base_q,1U<<24, calibration_model::BASE_Q,engine_base_q,context_base_q);
    // const string  engine_path_base_q{"../../backbone_q.engine"};
    // parseEngineModel(engine_path_base_q,engine_base_q,context_base_q);
    // parseOnnxModel(model_path_base_m,1U<<24,engine_base_m,context_base_m);
    parseOnnxModelINT8(model_path_base_m,1U<<24, calibration_model::MEMORIZE,engine_base_m,context_base_m);
    // const string  engine_path_base_m{"../../memorize.engine"};
    // parseEngineModel(engine_path_base_m,engine_base_m,context_base_m);

    // const string  engine_path_head{"../../head.engine"};
    // parseEngineModel(engine_path_head,engine_base_m,context_base_m);
    
    parseOnnxModel(model_path_head,1U<<24,engine_head,context_head);


    buffers_base_q.reserve(engine_base_q->getNbBindings());
    buffers_base_m.reserve(engine_base_m->getNbBindings());
    buffers_head.reserve(engine_head->getNbBindings());

    cout<<"------------------------------"<<endl;
    for (size_t i = 0; i < engine_base_q->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine_base_q->getBindingDimensions(i)) * batch_size * sizeof(float);
        // cudaMalloc(&buffers_base_q[i], binding_size);
        cudaMallocManaged(&buffers_base_q[i], binding_size);
        std::cout<<engine_base_q->getBindingName(i)<<std::endl;
        if (engine_base_q->bindingIsInput(i))
        {
            
            input_dims_base_q.emplace_back(engine_base_q->getBindingDimensions(i));
        }
        else
        {
            output_dims_base_q.emplace_back(engine_base_q->getBindingDimensions(i));
        }
    }
    cout<<"------------------------------"<<endl;
    for (size_t i = 0; i < engine_base_m->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine_base_m->getBindingDimensions(i)) * batch_size * sizeof(float);
        // cudaMalloc(&buffers_base_m[i], binding_size);
        cudaMallocManaged(&buffers_base_m[i], binding_size);
        std::cout<<engine_base_m->getBindingName(i)<<std::endl;
        if (engine_base_m->bindingIsInput(i))
        {
            
            input_dims_base_m.emplace_back(engine_base_m->getBindingDimensions(i));
        }
        else
        {
            output_dims_base_m.emplace_back(engine_base_m->getBindingDimensions(i));
        }
    }
    cout<<"------------------------------"<<endl;
    for (size_t i = 0; i < engine_head->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine_head->getBindingDimensions(i)) * batch_size * sizeof(float);
        cudaMalloc(&buffers_head[i], binding_size);
        std::cout<<engine_head->getBindingName(i)<<std::endl;
        if (engine_head->bindingIsInput(i))
        {
            input_dims_head.emplace_back(engine_head->getBindingDimensions(i));
        }
        else
        {
            output_dims_head.emplace_back(engine_head->getBindingDimensions(i));
        }
    }
    cout<<"------------------------------"<<endl;
    
    cudaStreamCreate(&stream_m);
    
    // cudaMallocHost((void **)&data_q, 1 * 3 * q_size * q_size*sizeof(float));
    // cudaMallocHost((void **)&data_m, 1 * 3 * m_size * m_size*sizeof(float));
    // cudaMallocHost((void **)&fg_bg_label_map, m_size * m_size*sizeof(float));

    cudaMallocHost((void **)&box, score_size*score_size*4*batch_size*sizeof(float));

    cudaMallocHost((void **)&score, score_size*score_size*batch_size*sizeof(float));

        // float box[score_size*score_size*4*batch_size];
    // float score[score_size*score_size*batch_size];



    // cudaHostAlloc((void **)&data_q, 1 * 3 * q_size * q_size*sizeof(float),  cudaHostAllocWriteCombined | cudaHostAllocMapped);
    // cudaHostAlloc((void **)&data_m, 1 * 3 * m_size * m_size*sizeof(float),  cudaHostAllocWriteCombined | cudaHostAllocMapped);
    // cudaHostAlloc((void **)&fg_bg_label_map, m_size * m_size*sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    
}

void stmtracker::init(Mat im, Rect2f state)
{
    // im: first frame for tracke
    // state: cv::Rect format constructed with xywh foramt 
    target_pos = (state.br()+state.tl())/2;
    target_sz = state.size();
    im_h = im.rows;
    im_w = im.cols;
    window = get_hann_win(Size(score_size,score_size));
    avg_chans = cv::mean(im);
    last_img = im;
    pscores.push_back(1.0f);
    cur_frame_idx = 1;
    float search_area = search_area_factor*search_area_factor*target_sz.area();
    target_scale = std::sqrt(search_area)/q_size;
    base_target_sz = target_sz/target_scale;

    return;
}

Rect2f stmtracker::update(Mat im)
{
    
    target_pos_prior = target_pos;
    target_sz_prior = target_sz;
    
    int fidx = cur_frame_idx;

    memorize();
    
    vector<int> indexes = select_representatives(fidx);

    // selelct pervious saved features for memeory 
    vector<void *> features;
    for(int i:indexes)
        features.push_back(all_memory_frame_feats[i]);

    Point2f new_target_pos;
    Size2f new_target_sz;
    
    track(im,features,new_target_pos,new_target_sz);

    float x1 = new_target_pos.x-new_target_sz.width/2;
    float y1 = new_target_pos.y-new_target_sz.height/2;    
    Rect2f track_rect = Rect2f(x1,y1,new_target_sz.width,new_target_sz.height);
    
    last_img = im;

    target_pos = new_target_pos;
    target_sz = new_target_sz;
    target_scale = std::sqrt( target_sz.area()/base_target_sz.area() ); 
    cur_frame_idx ++;

    
    
    return track_rect;
    
}
void stmtracker::track(Mat im_q,vector<void *> &features,Point2f &new_target_pos, Size2f &new_target_sz)
{
    Mat im_q_crop;
   
    get_crop_single(im_q,target_pos_prior,target_scale,q_size,avg_chans,im_q_crop,scale_q);
    
    // float data_q[1 * 3 * q_size * q_size];
    int data_idx = 0;
    data_q = (float * )buffers_base_q[0];
    // #define data_qq buffers_base_q[0]
    for (int i = 0; i < im_q_crop.rows; ++i)
    {
        uchar* pixel = im_q_crop.ptr<uchar>(i);  // point to first color in row
        for (int j = 0; j < im_q_crop.cols; ++j)
        {
            data_q[data_idx] = *pixel++;
            data_q[data_idx+q_size*q_size] = *pixel++;
            data_q[data_idx+2*q_size*q_size] = *pixel++;
            data_idx++;
        }
    }
    
    // cudaMemcpyAsync(buffers_base_q[0], data_q, batch_size * 3 * q_size * q_size * sizeof(float), cudaMemcpyHostToDevice);

    // context_base_q->enqueue(batch_size,buffers_base_q.data(),0,nullptr);
    context_base_q->enqueueV2(buffers_base_q.data(),0,nullptr);
    // context_base_q->executeV2(buffers_base_q.data());

    // now buffers_base_q[1] contains fq
    #define FQ_SIZE 512*25*25
    //-------------------------------------------------------------------------------------------------------------
    // the next two line is very important!! the first line is using memcpy that have cost, but using just pointers can solve our problem 
    // cudaMemcpy(buffers_head[1],buffers_base_q[1],batch_size*FQ_SIZE*sizeof(float),cudaMemcpyDeviceToDevice);
    buffers_head[6] = buffers_base_q[1]; // this is very better than pervius line!!
    //-------------------------------------------------------------------------------------------------------------
    // auto end = std::chrono::system_clock::now();
    
    int mem_step = score_size*score_size;
    
    for (int i = 0;i <6;i++)
        buffers_head[i] = features[i]; // 0 to 5 belong to fm1-6
    
    context_head->enqueueV2(buffers_head.data(),0,nullptr);

    // cudaDeviceSynchronize();

    auto start = std::chrono::system_clock::now();

    cudaMemcpyAsync(box, buffers_head[7],getSizeByDim(output_dims_head[0])*batch_size*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(score, buffers_head[8],getSizeByDim(output_dims_head[1])*batch_size*sizeof(float),cudaMemcpyDeviceToHost);
    
    cudaStreamSynchronize(0);
    vector<vector<float>> box_wh = xyxy2cxywh(box);

    vector<float> pscore;
    vector<float> penalty;
    _postprocess_score(score,box_wh,pscore,penalty);

    auto max_pscore_it = std::max_element(pscore.begin(),pscore.end());
    int best_pscore_id = distance(pscore.begin(), max_pscore_it);
    
    _postprocess_box(score[best_pscore_id],box_wh[best_pscore_id],penalty[best_pscore_id],new_target_pos,new_target_sz);
    auto end = std::chrono::system_clock::now();
    // std::cout <<"post process " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
    pscores.push_back(score[best_pscore_id]);

}

void stmtracker::_postprocess_box(float score_best,vector<float> box_best,float penalty_best,Point2f &new_target_pos,Size2f &new_target_sz)
{
    vector<float> pred_in_crop = {  box_best[0]/scale_q,
                                    box_best[1]/scale_q,
                                    box_best[2]/scale_q,
                                    box_best[3]/scale_q
                                    };

    float lr = penalty_best*score_best*test_lr;

    float res_x = pred_in_crop[0] + target_pos_prior.x - (q_size/2)/scale_q;
    float res_y = pred_in_crop[1] + target_pos_prior.y - (q_size/2)/scale_q;

    float res_w = target_sz_prior.width*(1-lr)+pred_in_crop[2]*lr;
    float res_h = target_sz_prior.height*(1-lr)+pred_in_crop[3]*lr;

    // do _restrict_box:
    res_x = std::max(0.0f,std::min( (float)im_w,res_x));
    res_y = std::max(0.0f,std::min( (float)im_h,res_y));

    res_w = std::max( (float)min_w,std::min((float)im_w,res_w));
    res_h = std::max( (float)min_h,std::min((float)im_h,res_h));

    new_target_pos = Point2f(res_x,res_y);
    new_target_sz = Size2f(res_w,res_h);
}

void stmtracker::_postprocess_score(float * score,const vector<vector<float>> &box_wh,vector<float> &pscore,vector<float> &penalty )
{
    pscore.clear();
    penalty.clear();

    Size2f target_sz_in_crop = target_sz_prior*scale_q;
    // scale penalty
    for(int i=0; i <625; i++)
    {
        float s_c = change(
                    sz(box_wh[i][2],box_wh[i][3])/
                    sz(target_sz_in_crop.width,target_sz_in_crop.height)
        );

        float r_c = change( (target_sz_in_crop.width/target_sz_in_crop.height) /
                            (box_wh[i][2]/box_wh[i][3]));

        float penalty_ = std::exp(-(r_c * s_c - 1) * penalty_k);
        float pscore_ = penalty_ * score[i];
        pscore_ = pscore_*(1-window_influence)+ window.at<float>(i/25,i%25)*window_influence;
        pscore.push_back(pscore_);
        penalty.push_back(penalty_);
    }

}

float stmtracker::change(float r)
{
    return std::max(r,1.0f/r);
}

float stmtracker::sz(float w,float h)
{
    float pad = (w+h)/2;
    float sz2 = (w+pad)*(h+pad);
    return std::sqrt(sz2);
}

vector<int> stmtracker::select_representatives(int cur_frame_idx)
{
    // TODO: consider pscore for appending index 
    std::vector<int> indexes;
    if (cur_frame_idx>num_segments)
    {
        indexes.push_back(0);
        int dur = (cur_frame_idx-1)/num_segments;

        for(int i =0;i<num_segments;i++)
            indexes.push_back(i*dur+dur/2+1);

        indexes.push_back(cur_frame_idx-1); 
    }
    else 
    {
        for (int i=0;i<cur_frame_idx;i++)
            indexes.push_back(i);

        while (indexes.size()<6)
        {
            indexes.push_back(cur_frame_idx-1);
        }
    }
    
    return indexes; 
}

void stmtracker::memorize()
{
    // const int m_siz{289};
    float scale_m = std::sqrt(target_sz_prior.area()/base_target_sz.area());
    Mat im_m_crop;
    float real_scale;
    // float data_m[1 * 3 * m_size * m_size];
    get_crop_single(last_img,target_pos,scale_m,m_size,avg_chans,im_m_crop,real_scale);

    int data_idx = 0;
    data_m = (float * ) buffers_base_m[0];
    for (int i = 0; i < im_m_crop.rows; ++i)
    {
        uchar* pixel = im_m_crop.ptr<uchar>(i);  // point to first color in row
        for (int j = 0; j < im_m_crop.cols; ++j)
        {
            data_m[data_idx] = *pixel++;
            data_m[data_idx+q_size*q_size] = *pixel++;
            data_m[data_idx+2*q_size*q_size] = *pixel++;
            data_idx++;
        }
    }

    
    int x1 = (m_size-1)/2 - target_sz_prior.width*real_scale/2;
    int y1 = (m_size-1)/2 - target_sz_prior.height*real_scale/2;
    int x2 = (m_size-1)/2 + target_sz_prior.width*real_scale/2;
    int y2 = (m_size-1)/2 + target_sz_prior.height*real_scale/2;
/*
    int xyxy[4];
    xyxy[0] = (m_size-1)/2 - target_sz_prior.width*real_scale/2;
    xyxy[1] = (m_size-1)/2 - target_sz_prior.height*real_scale/2;
    xyxy[2] = (m_size-1)/2 + target_sz_prior.width*real_scale/2;
    xyxy[3] = (m_size-1)/2 + target_sz_prior.height*real_scale/2;

    dim3 dim_m(m_size,m_size);
    dim3 b(1);
    int * d_xyxy;
    cudaMalloc((void **)&d_xyxy, 4*sizeof(int));
    cudaMemcpyAsync(xyxy,d_xyxy,4*sizeof(int),cudaMemcpyHostToDevice);
*/
    // fill_m <<<dim_m,32 >>>(buffers_base_m[1],d_xyxy);


    // float fg_bg_label_map[m_size * m_size] ;
    fg_bg_label_map = (float *) buffers_base_m[1];

    for (int i = 0; i < m_size; i++)
        for (int j = 0; j < m_size ; j++)
        {
            if( j> x1-1 && j <x2+1 && i> y1-1 && i <y2+1 )
                fg_bg_label_map[m_size*i+j ] = 1.0;
            else 
                fg_bg_label_map[m_size*i+j ] = 0.0;
        }


    // cudaMemcpy(buffers_base_m[0], data, batch_size * 3 * m_size * m_size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaStream_t stream_m;
    // cudaStreamCreate(&stream_m);

    // cudaMemcpyAsync(buffers_base_m[0], data_m, batch_size * 3 * m_size * m_size * sizeof(float), cudaMemcpyHostToDevice,stream_m);


    // cudaMemcpyAsync(buffers_base_m[0], data_m, batch_size * 3 * m_size * m_size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(buffers_base_m[1], fg_bg_label_map, batch_size * 1 * m_size * m_size * sizeof(float), cudaMemcpyHostToDevice);

    // cudaMemcpyAsync(buffers_base_m[1], fg_bg_label_map, batch_size * 1 * m_size * m_size * sizeof(float), cudaMemcpyHostToDevice,stream_m);

    // cudaMemcpyAsync(buffers_base_m[1], fg_bg_label_map, batch_size * 1 * m_size * m_size * sizeof(float), cudaMemcpyHostToDevice);

    void * temp_ptr;
    cudaMalloc(&temp_ptr, getSizeByDim(output_dims_base_m[0])* sizeof(float));
    buffers_base_m[2] = temp_ptr;
    // context_base_m->enqueueV2(buffers_base_m.data(),0,nullptr);
    context_base_m->enqueueV2(buffers_base_m.data(),stream_m,nullptr);
    // context_base_m->executeV2(buffers_base_m.data());
    

    // cudaMalloc(&temp_ptr, getSizeByDim(output_dims_base_m[0])* sizeof(float));
    // cudaMemcpyAsync(temp_ptr,(float *) buffers_base_m[2],getSizeByDim(output_dims_base_m[0])*sizeof(float),cudaMemcpyDeviceToDevice,stream_m);
    // cudaMemcpyAsync(temp_ptr,(float *) buffers_base_m[2],getSizeByDim(output_dims_base_m[0])*sizeof(float),cudaMemcpyDeviceToDevice);

    all_memory_frame_feats.push_back(temp_ptr);
    return;
}

stmtracker::~stmtracker()
{
    for (void * buf : buffers_base_q)
        cudaFree(buf);

    for (void * buf : buffers_base_m)
        cudaFree(buf);

    for (void * buf : buffers_head)
        cudaFree(buf);

    for (void * buf :all_memory_frame_feats)
        cudaFree(buf);

    cudaStreamDestroy(stream_m);

    // cudaFreeHost(data_q);
    // cudaFreeHost(data_m);
    // cudaFreeHost(fg_bg_label_map);
    cudaFreeHost(box);
    cudaFreeHost(score);

}




