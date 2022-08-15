#include "stmtracker.hpp"
// #include "utils.hpp"

#define batch_size 1
stmtracker::stmtracker(/* args */)
{
    
    // parseOnnxModel(model_path_base_q,engine_base_q,context_base_q);
    const string  engine_path_base_q{"../../backbone_q.engine"};
    parseEngineModel(engine_path_base_q,engine_base_q,context_base_q);
    // parseOnnxModel(model_path_base_m,engine_base_m,context_base_m);
    const string  engine_path_base_m{"../../memorize.engine"};
    parseEngineModel(engine_path_base_m,engine_base_m,context_base_m);
    
    parseOnnxModel(model_path_head,engine_head,context_head);
    
    // const string  head_path{"../../head.engine"};
    // serializeOnnx2engine(engine_head,head_path);

    buffers_base_q.reserve(engine_base_q->getNbBindings());
    buffers_base_m.reserve(engine_base_m->getNbBindings());
    buffers_head.reserve(engine_head->getNbBindings());

    
    cout<<"------------------------------"<<endl;
    for (size_t i = 0; i < engine_base_q->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine_base_q->getBindingDimensions(i)) * batch_size * sizeof(float);
        cudaMalloc(&buffers_base_q[i], binding_size);
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
        cudaMalloc(&buffers_base_m[i], binding_size);
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

    
    cudaStreamCreate(&stream_m);
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
    //  
    auto start = std::chrono::system_clock::now();
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

    // cout<<track_rect<<endl;
    
    last_img = im;

    target_pos = new_target_pos;
    target_sz = new_target_sz;
    target_scale = std::sqrt( target_sz.area()/base_target_sz.area() ); 
    cur_frame_idx ++;

    auto end = std::chrono::system_clock::now();
    std::cout <<"update " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    
    return track_rect;
    
}
void stmtracker::track(Mat im_q,vector<void *> &features,Point2f &new_target_pos, Size2f &new_target_sz)
{
    Mat im_q_crop;
   
    get_crop_single(im_q,target_pos_prior,target_scale,q_size,avg_chans,im_q_crop,scale_q);
    
    float data[1 * 3 * q_size * q_size];
    int data_idx = 0;
    
    for (int i = 0; i < im_q_crop.rows; ++i)
    {
        cv::Vec3b* pixel = im_q_crop.ptr<cv::Vec3b>(i); // point to first pixel in row
        for (int j = 0; j < im_q_crop.cols; ++j)
        {
            data[data_idx] = pixel[j][0];
            data[data_idx+m_size * m_size]  = pixel[j][1];
            data[data_idx+2*m_size * m_size] = pixel[j][2]; 
            data_idx++;
        }
    }
   
    cudaMemcpy(buffers_base_q[0], data, batch_size * 3 * q_size * q_size * sizeof(float), cudaMemcpyHostToDevice);
    // context_base_q->enqueue(batch_size,buffers_base_q.data(),0,nullptr);
    context_base_q->enqueueV2(buffers_base_q.data(),0,nullptr);
    // context_base_q->executeV2(buffers_base_q.data());

    // now buffers_base_q[1] contains fq
    #define FQ_SIZE 512*25*25
    auto start = std::chrono::system_clock::now();
    //-------------------------------------------------------------------------------------------------------------
    // the next two line is very important!! the first line is using memcpy that have cost, but using just pointers can solve our problem 
    // cudaMemcpy(buffers_head[1],buffers_base_q[1],batch_size*FQ_SIZE*sizeof(float),cudaMemcpyDeviceToDevice);
    buffers_head[6] = buffers_base_q[1]; // this is very better than pervius line!!
    //-------------------------------------------------------------------------------------------------------------
    auto end = std::chrono::system_clock::now();
    
    int mem_step = score_size*score_size;
    
    // TODO: change this strange line of code into the zero copy code 

    // for (int i = 0;i <512;i++)
    // {
    //     for (int j = 0;j <6;j++)
    //     {
    //         float *dst = static_cast<float *>(buffers_head[0]);
    //         dst += (6*i+j)*mem_step;
    //         float *src = static_cast<float *>(features[j]);
    //         src += i*mem_step;
    //         cudaMemcpy( (void *) dst,
    //                     (void *) src,
    //                     batch_size*mem_step*sizeof(float),cudaMemcpyDeviceToDevice);
    //     }
    // }

    for (int j = 0;j <6;j++)
        buffers_head[j] = features[j]; // 0 to 5 belong to fm1-6
    
    std::cout <<"memcpy " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
    context_head->enqueueV2(buffers_head.data(),0,nullptr);


    vector<float> box(getSizeByDim(output_dims_head[0])*batch_size);
    cudaMemcpy(box.data(), buffers_head[7],box.size()*sizeof(float),cudaMemcpyDeviceToHost);

    // vector<float> cls(getSizeByDim(output_dims_head[1])*batch_size);
    // cudaMemcpy(cls.data(), buffers_head[3],cls.size()*sizeof(float),cudaMemcpyDeviceToHost);

    // vector<float> ctr(getSizeByDim(output_dims_head[2])*batch_size);
    // cudaMemcpy(ctr.data(), buffers_head[4],ctr.size()*sizeof(float),cudaMemcpyDeviceToHost);

    vector<float> score(getSizeByDim(output_dims_head[1])*batch_size);
    cudaMemcpy(score.data(), buffers_head[8],score.size()*sizeof(float),cudaMemcpyDeviceToHost);


    // vector<float> tempp(getSizeByDim(output_dims_head[2])*batch_size);
    // cudaMemcpy(tempp.data(), buffers_head[3],tempp.size()*sizeof(float),cudaMemcpyDeviceToHost);

    vector<vector<float>> box_wh = xyxy2cxywh(box);

    vector<float> pscore;
    vector<float> penalty;
    _postprocess_score(score,box_wh,pscore,penalty);

    auto max_pscore_it = std::max_element(pscore.begin(),pscore.end());
    int best_pscore_id = distance(pscore.begin(), max_pscore_it);
    
    _postprocess_box(score[best_pscore_id],box_wh[best_pscore_id],penalty[best_pscore_id],new_target_pos,new_target_sz);

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
    // cout<<lr<<endl;
    float res_x = pred_in_crop[0] + target_pos_prior.x - (q_size/2)/scale_q;
    float res_y = pred_in_crop[1] + target_pos_prior.y - (q_size/2)/scale_q;

    float res_w = target_sz_prior.width*(1-lr)+pred_in_crop[2]*lr;
    float res_h = target_sz_prior.height*(1-lr)+pred_in_crop[3]*lr;

    // std::cout<<"res_x: "<<res_x<<" res_y "<<res_y<<" res_w "<<res_w<<" res_h "<<res_h<<endl;
    // do _restrict_box:

    res_x = std::max(0.0f,std::min( (float)im_w,res_x));
    res_y = std::max(0.0f,std::min( (float)im_h,res_y));

    res_w = std::max( (float)min_w,std::min((float)im_w,res_w));
    res_h = std::max( (float)min_h,std::min((float)im_h,res_h));

    new_target_pos = Point2f(res_x,res_y);
    new_target_sz = Size2f(res_w,res_h);
}

void stmtracker::_postprocess_score(vector<float> score,vector<vector<float>> box_wh,vector<float> &pscore,vector<float> &penalty )
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
    float data[1 * 3 * m_size * m_size];
    get_crop_single(last_img,target_pos,scale_m,m_size,avg_chans,im_m_crop,real_scale);

    int data_idx = 0;
    for (int i = 0; i < im_m_crop.rows; ++i)
    {
        cv::Vec3b* pixel = im_m_crop.ptr<cv::Vec3b>(i); // point to first pixel in row
        for (int j = 0; j < im_m_crop.cols; ++j)
        {
            data[data_idx] = pixel[j][0];
            data[data_idx+m_size * m_size]  = pixel[j][1];
            data[data_idx+2*m_size * m_size] = pixel[j][2]; 
            data_idx++;
        }
    }

    int x1 = (m_size-1)/2 - target_sz_prior.width*real_scale/2;
    int y1 = (m_size-1)/2 - target_sz_prior.height*real_scale/2;
    int x2 = (m_size-1)/2 + target_sz_prior.width*real_scale/2;
    int y2 = (m_size-1)/2 + target_sz_prior.height*real_scale/2;

    float fg_bg_label_map[m_size * m_size] ;
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
    cudaMemcpyAsync(buffers_base_m[0], data, batch_size * 3 * m_size * m_size * sizeof(float), cudaMemcpyHostToDevice,stream_m);
    // cudaMemcpy(buffers_base_m[1], fg_bg_label_map, batch_size * 1 * m_size * m_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(buffers_base_m[1], fg_bg_label_map, batch_size * 1 * m_size * m_size * sizeof(float), cudaMemcpyHostToDevice,stream_m);
    // context_base_m->enqueue(batch_size,buffers_base_m.data(),0,nullptr);
    context_base_m->enqueueV2(buffers_base_m.data(),stream_m,nullptr);
    // context_base_m->executeV2(buffers_base_m.data());
    void * temp_ptr;
    // cudaMalloc(&temp_ptr, getSizeByDim(output_dims_base_m[0])* sizeof(float));

    cudaMalloc(&temp_ptr, getSizeByDim(output_dims_base_m[0])* sizeof(float));
    cudaMemcpyAsync(temp_ptr,(float *) buffers_base_m[2],getSizeByDim(output_dims_base_m[0])*sizeof(float),cudaMemcpyDeviceToDevice,stream_m);
    // cudaMemcpyAsync(temp_ptr,(float *) buffers_base_m[2],getSizeByDim(output_dims_base_m[0])*sizeof(float),cudaMemcpyDeviceToDevice);

    all_memory_frame_feats.push_back(temp_ptr);
    return;
}

void stmtracker::get_crop_single(Mat & im,Point2f target_pos_,
                                float target_scale,int output_sz,Scalar avg_chans,
                                Mat &im_patch,float &real_scale) // these are output 
{
    // reversed pos!! 
    // pos is target_pos

    Point2i posl = Point2i(target_pos_);
    float sample_sz = target_scale*output_sz;    
    // resize_factor = np.min(sample_sz.astype(np.float) / output_sz.astype(np.float)).item()
    // float resize_factor = std::min(sample_sz,static_cast<float>(output_sz));
    float resize_factor = sample_sz/output_sz;
    
    int df = std::max(static_cast<int>(resize_factor-0.1),1);
    Mat im2;

    // auto start = std::chrono::system_clock::now();

    
    if (df > 1)
    {
        // Point2i os = Point2i(static_cast<int>(target_pos_.x) % df,static_cast<int>(target_pos_.y) % df);
        Point2i os = Point2i( posl.x % df, posl.y % df);
        std::vector<uchar> im_rows;
        std::vector< std::vector<uchar> > im_cols;
        posl = Point2i((posl.x-os.x)/df , (posl.y-os.y)/df);

        int channels = im.channels();

        int nRows = im.rows;
        int nCols = im.cols * channels;
        int i,j;
        uchar* p;
        std::vector<uchar> pixel_vec;
        int reduced_col_sz = 0;
        int reduced_row_sz = 0;
        for( i = os.x; i < nRows; i+=df)
        {
            p = im.ptr<uchar>(i);
            // im_cols.clear(); 
            reduced_col_sz = 0;
            for ( j = os.y*3; j < nCols;j+=3*df)
            {
                pixel_vec.push_back(p[j]);
                pixel_vec.push_back(p[j+1]);
                pixel_vec.push_back(p[j+2]);
                reduced_col_sz ++;
            }
            reduced_row_sz++;
            // im_cols.push_back(im_rows);
        }

        im2 = cv::Mat(reduced_row_sz, reduced_col_sz, CV_8UC3,pixel_vec.data()).clone();
    }
    else
    {
        im2 = im;
    }

    float sz = sample_sz/df;
    
    int szl = std::max(static_cast<int>(std::round(sz)) ,2);

    Point2i tl = Point2i(posl.x-(szl-1)/2,posl.y-(szl-1)/2);
    Point2i br = Point2i(posl.x+szl/2+1,posl.y+szl/2+1);

    float M_13 = tl.x;
    float M_23 = tl.y;
    float M_11 = (br.x-M_13)/(output_sz-1);
    float M_22 = (br.y-M_23)/(output_sz-1);

    float arr[2][3] = { 
                        {M_11,0,M_13},
                        {0,M_22,M_23}
                        };

    Mat mat2x3 = Mat(2,3,CV_32FC1 ,arr);

    cv::warpAffine( im2,
                    im_patch,
                    mat2x3,
                    Size(output_sz,output_sz),
                    INTER_LINEAR | WARP_INVERSE_MAP,
                    BORDER_CONSTANT,
                    avg_chans
    ); 	
    real_scale = static_cast<float>(output_sz)/((br.x-tl.x+1)*df) ;   
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
    
}




