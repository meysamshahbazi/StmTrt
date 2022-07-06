#include "stmtracker.hpp"
// #include "utils.hpp"

#define batch_size 1
stmtracker::stmtracker(/* args */)
{
    /*
    parseOnnxModel(model_path_base_q,engine_base_q,context_base_q);
    parseOnnxModel(model_path_base_m,engine_base_m,context_base_m);
    parseOnnxModel(model_path_head,engine_head,context_head);

    buffers_base_q.reserve(engine_base_q->getNbBindings());
    buffers_base_m.reserve(engine_base_m->getNbBindings());
    buffers_head.reserve(engine_head->getNbBindings());

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
    */

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

    // std::cout<<search_area<<endl;
    // std::cout<<target_scale<<endl;
    // std::cout<<base_target_sz<<endl;
    

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
    
}




