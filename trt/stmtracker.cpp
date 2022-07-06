#include "stmtracker.hpp"
// #include "utils.hpp"

#define batch_size 1
stmtracker::stmtracker(/* args */)
{
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




