
#ifndef ONEFLOW_GRAPH_HELPER_HPP
#define ONEFLOW_GRAPH_HELPER_HPP

#include <map>

namespace oneflow{

    enum class BldTskGphMtdType{
        Boxing,
        One2One,
        SelectOneSourceToSoleSink,
        Scatter2LocalAdd,
        Scatter2GlobalAdd,
        LocalAdd2GlobalAdd,
        GlobalAdd2Gather,
        Unknown
    };

    class GraphHelper{
    public:
        static GraphHelper& get() {
            static GraphHelper instance;
            return instance;
        }

        BldTskGphMtdType GetMtdType(const LogicalNode* src_node, const LogicalNode* dst_node){
            if(dst_node->TypeName()=="MdSave"){
                if (dst_node->parallel_desc()->parallel_num() == 1) {
                    return BldTskGphMtdType::SelectOneSourceToSoleSink;
                } else {
                    return BldTskGphMtdType::One2One;
                }
            }else if(dst_node->TypeName()=="NormalMdUpdt"){
                if (dst_node->parallel_desc()->policy() == kDataParallel) {
                    return BldTskGphMtdType::Boxing;
                } else if (dst_node->parallel_desc()->policy() == kModelParallel) {
                    return BldTskGphMtdType::One2One;
                }else{
                    return BldTskGphMtdType::Unknown;
                }
            }else{
                std::shared_ptr<const ParallelDesc> src_pd = src_node->parallel_desc();
                std::shared_ptr<const ParallelDesc> dst_pd = dst_node->parallel_desc();
                if (src_pd->parallel_num() == 1 && dst_pd->parallel_num() == 1) {
                    return BldTskGphMtdType::One2One;
                }

                std::string key = src_node->TypeName() + dst_node->TypeName();
                BldTskGphMtdType mthd_type = GetMtdType(key);
                if(mthd_type!=BldTskGphMtdType::Unknown){
                    return mthd_type;
                }

                if (src_pd->parallel_num() == dst_pd->parallel_num()) {
                    if (src_pd->policy() == kDataParallel && dst_pd->policy() == kDataParallel) {
                        return BldTskGphMtdType::One2One;
                    } else if (src_pd->policy() == kModelParallel && dst_pd->policy() == kModelParallel
                               && IsModelParallel121(src_node, dst_node)) {
                        return BldTskGphMtdType::One2One;
                    }
                }

                return BldTskGphMtdType::Boxing;
            }
        }

        void RegisterMtdType(const std::string& key, BldTskGphMtdType type){
            if(type==BldTskGphMtdType::Unknown)
                return;

            map_.emplace(key, type);
        }

    private:
        GraphHelper() = default;
        GraphHelper(const GraphHelper&) = delete;
        GraphHelper(GraphHelper&&) = delete;

        bool IsModelParallel121(const LogicalNode* src_node, const LogicalNode* dst_node) {
            return src_node->main_model_parallel() == dst_node->main_model_parallel();
        }

        BldTskGphMtdType GetMtdType(const std::string& key){
            auto it = map_.find(key);
            if(it!=map_.end())
                return it->second;

            return BldTskGphMtdType::Unknown;
        }

        std::map<std::string, BldTskGphMtdType> map_;
    };
}

#endif //ONEFLOW_GRAPH_HELPER_HPP
