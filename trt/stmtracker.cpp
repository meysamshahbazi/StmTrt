#include "stmtracker.hpp"

#define batch_size 1



stmtracker::stmtracker() {

    // setImageParams();
    im_width = 1920;
    im_hight = 1080; 
    load_trt_engines();
    // create_trt_buffer();
    // cudaStreamCreate(&stream_m);
    // box = new float[score_size*score_size*4];
    // score = new float[score_size*score_size];

    // // cudaMallocHost((void **)&box, score_size*score_size*4*batch_size*sizeof(float));
    // // cudaMallocHost((void **)&score, score_size*score_size*batch_size*sizeof(float));


    // // cudaMallocHost((void **)&box, score_size*score_size*4*batch_size*sizeof(float));
    // // cudaMallocHost((void **)&score, score_size*score_size*batch_size*sizeof(float));
    // cudaMalloc((void **)&im_q_crop_ptr, q_size*q_size*batch_size*sizeof(uchar4));
    // cudaMalloc((void **)&im_m_crop_ptr, m_size*m_size*batch_size*sizeof(uchar4));
    // createWindow();
}

void stmtracker::init(cv::Mat fd, const cv::Rect2f state_)
{
    state = state_; 
    target_pos =/*  getCenter(state); // */ (state.br() + state.tl()) / 2;
    target_sz = state.size();
    last_fd = fd;
    pscores.clear();
    
    pscores.push_back(1.0f);
    cur_frame_idx = 1;
    float search_area = search_area_factor*search_area_factor*target_sz.area();
    target_scale = std::sqrt(search_area)/q_size;
    base_target_sz = target_sz/target_scale;
    for (void * buf :all_memory_frame_feats)
        cudaFree(buf);

    all_memory_frame_feats.clear();
}


Rect2f stmtracker::update(cv::Mat fd) {
    target_pos_prior = target_pos;
    target_sz_prior = target_sz;
    
    int fidx = cur_frame_idx;

    // last_fd = fd; // remove this!
    memorize();
    cudaStreamSynchronize(0);
    vector<int> indexes = select_representatives(fidx);

    // selelct pervious saved features for memeory 
    vector<void *> features;
    for(int i:indexes)
        features.push_back(all_memory_frame_feats[i]);

    Point2f new_target_pos;
    Size2f new_target_sz;
    
    track(fd,features,new_target_pos,new_target_sz);

    float x1 = new_target_pos.x-new_target_sz.width/2;
    float y1 = new_target_pos.y-new_target_sz.height/2;    
    Rect2f track_rect = Rect2f(x1,y1,new_target_sz.width,new_target_sz.height);
    
    // last_img = im;
    last_fd = fd;

    target_pos = new_target_pos;
    target_sz = new_target_sz;
    target_scale = std::sqrt( target_sz.area()/base_target_sz.area() ); 
    cur_frame_idx ++;
    return track_rect;
    // return state;
}


void stmtracker::track(cv::Mat fd, std::vector<void *> &features, cv::Point2f &new_target_pos, cv::Size2f &new_target_sz)
{
    /* cv::Mat im_q */
    // Mat im_q_crop;
       
    data_q = (float * )buffers_base_q[0];
    get_crop_single2(fd, target_pos_prior,target_scale, data_q, im_q_crop_ptr, scale_q,
                                /* cudaStream_t stream */ 0);
    cudaStreamSynchronize(0);
    // float data_q[1 * 3 * q_size * q_size];
    int data_idx = 0;
    // #define data_qq buffers_base_q[0]

    

    // context_base_q->enqueue(batch_size,buffers_base_q.data(),0,nullptr);
    context_base_q->enqueueV2(buffers_base_q.data(),0,nullptr);
    // context_base_q->executeV2(buffers_base_q.data());
    cudaStreamSynchronize(0);
    // now buffers_base_q[1] contains fq
    #define FQ_SIZE 20*25*25
    //-------------------------------------------------------------------------------------------------------------
    // the next two line is very important!! the first line is using memcpy that have cost, but using just pointers can solve our problem 
    // cudaMemcpy(buffers_head[1],buffers_base_q[1],batch_size*FQ_SIZE*sizeof(float),cudaMemcpyDeviceToDevice);
    buffers_head[6] = buffers_base_q[1]; // this is very better than pervius line!!
    //-------------------------------------------------------------------------------------------------------------
    // auto end = std::chrono::system_clock::now();
    
    
    for (int i = 0;i <6;i++)
        buffers_head[i] = features[i]; // 0 to 5 belong to fm1-6
    
    context_head->enqueueV2(buffers_head.data(),0,nullptr);

    cudaDeviceSynchronize();

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
    res_x = std::max(0.0f, std::min( (float)im_width,res_x));
    res_y = std::max(0.0f, std::min( (float)im_hight,res_y));

    res_w = std::max((float)min_w, std::min((float)im_width, res_w));
    res_h = std::max((float)min_h, std::min((float)im_hight, res_h));

    new_target_pos = cv::Point2f(res_x,res_y);
    new_target_sz = cv::Size2f(res_w,res_h);
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
        pscore_ = pscore_*(1-window_influence)+ window.at<float>(i/25,i%25)*window_influence; // TODO!
        pscore.push_back(pscore_);
        penalty.push_back(penalty_);
    }
}

float stmtracker::change(float r)
{
    return std::max(r,1.0f/r);
}

float stmtracker::sz(float w, float h)
{
    float pad = (w+h)/2;
    float sz2 = (w+pad)*(h+pad);
    return std::sqrt(sz2);
}

void stmtracker::memorize() {
    float scale_m = std::sqrt(target_sz_prior.area()/base_target_sz.area());
    float real_scale;
    data_m = (float *) buffers_base_m[0];
    
    get_crop_single2(last_fd, target_pos, scale_m, data_m, 
                                im_m_crop_ptr, real_scale,
                                /* cudaStream_t stream */0); // these are output 



    int x1 = (m_size-1)/2 - target_sz_prior.width*real_scale/2;
    int y1 = (m_size-1)/2 - target_sz_prior.height*real_scale/2;
    int x2 = (m_size-1)/2 + target_sz_prior.width*real_scale/2;
    int y2 = (m_size-1)/2 + target_sz_prior.height*real_scale/2;


    // float fg_bg_label_map[m_size * m_size];
    fg_bg_label_map = (float *) buffers_base_m[1];
    cv::Rect2i bb_(x1, y1, x2 - x1, y2 - y1);
    create_fg_bg_label_map(fg_bg_label_map, bb_,/*cudaStream_t stream */ 0);

    void * temp_ptr;
    cudaMalloc(&temp_ptr, getSizeByDim(output_dims_base_m[0])* sizeof(float));
    buffers_base_m[2] = temp_ptr;

    context_base_m->enqueueV2(buffers_base_m.data(), 0, nullptr);

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

    // cudaFreeHost(box);
    // cudaFreeHost(score);
}



void stmtracker::load_trt_engines() {
// #define GEN_ENGINE_FROM_ONNX
// #define LOAD_FROM_ONNX
#define LOAD_FROM_ENGINE

#ifdef GEN_ENGINE_FROM_ONNX
    gen_engine_from_onnx();
#endif

#ifdef LOAD_FROM_ONNX
    parseOnnxModel(model_path_base_q,1U<<24,engine_base_q,context_base_q);
    parseOnnxModel(model_path_base_m,1U<<24,engine_base_m,context_base_m);
    
#endif

#ifdef LOAD_FROM_ENGINE
    const string  engine_path_base_q{"../../eff-onnx/backbone_q.engine"};
    parseEngineModel(engine_path_base_q,engine_base_q,context_base_q);
    const string  engine_path_base_m{"../../eff-onnx/memorize.engine"};
    parseEngineModel(engine_path_base_m,engine_base_m,context_base_m);

    const string  engine_path_head{"../../eff-onnx/head.engine"};
    parseEngineModel(engine_path_head,engine_head,context_head);

    // for head there is a bug in engine file!
    // parseOnnxModel(model_path_head, 1U<<31, engine_head,context_head);
#endif
}

void stmtracker::create_trt_buffer() {
    buffers_base_q.reserve(engine_base_q->getNbBindings());
    buffers_base_m.reserve(engine_base_m->getNbBindings());
    buffers_head.reserve(engine_head->getNbBindings());

    for (size_t i = 0; i < engine_base_q->getNbBindings(); ++i) {
        auto binding_size = getSizeByDim(engine_base_q->getBindingDimensions(i)) * batch_size * sizeof(float);
        cudaMalloc(&buffers_base_q[i], binding_size);
        // cudaMallocManaged(&buffers_base_q[i], binding_size);
        if (engine_base_q->bindingIsInput(i)) {
            input_dims_base_q.emplace_back(engine_base_q->getBindingDimensions(i));
        }
        else {
            output_dims_base_q.emplace_back(engine_base_q->getBindingDimensions(i));
        }
    }

    for (size_t i = 0; i < engine_base_m->getNbBindings(); ++i) {
        auto binding_size = getSizeByDim(engine_base_m->getBindingDimensions(i)) * batch_size * sizeof(float);
        cudaMalloc(&buffers_base_m[i], binding_size);
        // cudaMallocManaged(&buffers_base_m[i], binding_size);
        if (engine_base_m->bindingIsInput(i)) {
            input_dims_base_m.emplace_back(engine_base_m->getBindingDimensions(i));
        }
        else {
            output_dims_base_m.emplace_back(engine_base_m->getBindingDimensions(i));
        }
    }

    for (size_t i = 0; i < engine_head->getNbBindings(); ++i) {
        auto binding_size = getSizeByDim(engine_head->getBindingDimensions(i)) * batch_size * sizeof(float);
        cudaMalloc(&buffers_head[i], binding_size);
        if (engine_head->bindingIsInput(i)) {
            input_dims_head.emplace_back(engine_head->getBindingDimensions(i));
        }
        else {
            output_dims_head.emplace_back(engine_head->getBindingDimensions(i));
        }
    }
}

vector<int> stmtracker::select_representatives(int cur_frame_idx)
{
    // TODO: consider pscore for appending index 
    std::vector<int> indexes;
    if (cur_frame_idx > num_segments) {
        indexes.push_back(0);
        int dur = (cur_frame_idx-1)/num_segments;

        for(int i =0;i<num_segments;i++)
            indexes.push_back(i*dur+dur/2+1);

        indexes.push_back(cur_frame_idx-1); 
    }
    else  {
        for (int i=0;i<cur_frame_idx;i++)
            indexes.push_back(i);

        while (indexes.size()<6)
            indexes.push_back(cur_frame_idx-1);
    }
    return indexes; 
}

void stmtracker::createWindow() {
    window = get_hann_win(cv::Size(score_size,score_size));
    float* h_window = new float[score_size*score_size];
    int window_idx = 0;
    for(int j = 0; j < score_size; j++) {
        for(int k = 0; k < score_size; k++) {
            h_window[window_idx] = window.at<float>(j,k);
            window_idx++;
        }
    }
    cudaMalloc(&d_window, score_size*score_size*sizeof(float));
    cudaMemcpy(d_window, h_window, score_size*score_size*sizeof(float),cudaMemcpyHostToDevice);
}

std::vector<vector<float>> stmtracker::xyxy2cxywh(float * box)
{
    std::vector<vector<float>> box_wh;
    for(int i = 0;i < 625; i++) {
        float cx = (box[4*i+0]+box[4*i+2])/2;
        float cy = (box[4*i+1]+box[4*i+3])/2;
        float w = box[4*i+2]-box[4*i+0]+1;
        float h = box[4*i+3]-box[4*i+1]+1;
        box_wh.push_back({cx,cy,w,h});
    } 
    return box_wh;
}

void stmtracker::gen_engine_from_onnx() {
    // saveEngineFile("../../eff-onnx/backbone_q.onnx","../../eff-onnx/backbone_q.engine");
    // saveEngineFile("../../eff-onnx/memorize.onnx","../../eff-onnx/memorize.engine");
    saveEngineFile("../../eff-onnx/head.onnx","../../eff-onnx/head.engine");
    return;
}