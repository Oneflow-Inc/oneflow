编译期的 C++ 

oneflow.h


## oneflow.cpp::FilterOpName2ParallelBlobConf<br>

FilterOpName2ParallelBlobConf 干了啥？<br>
获取当前 job 中需要内存复用的 OpBlob 的信息，将当前 job 的 (op_conf，parallel_blob_conf）信息添加到 op_name2parallel_blob_conf 集合里。<br>

```.cpp
/**
FilterOpName2ParallelBlobConf：获取当前 job 中需要内存复用的 OpBlob 的信息，将当前 job 的 (op_conf，parallel_blob_conf）信息添加到 op_name2parallel_blob_conf 集合里。
判断当前 job 的 op_conf.op_type_case() 是否在匹配集合 match 对象 （OperatorConf::OpTypeCase 类型）里：
如果有，获取当前 job 中需要内存复用的 OpBlob 的信息。再判断当前 job 的 op_conf 是否在 op_name2parallel_blob_conf 集合里：
如果有，则校验两者的 parallel_blob_conf 是否相等；
如果没有，则 op_name2parallel_blob_conf 添加当前 job 的信息。

*/
void FilterOpName2ParallelBlobConf(
    const HashSet<OperatorConf::OpTypeCase>& match, const std::vector<std::shared_ptr<Job>>& jobs,
    HashMap<std::string, ParallelBlobConf>* op_name2parallel_blob_conf) {
  FOR_RANGE(int64_t, job_id, 0, jobs.size()) {
    JobBuilder job_builder(jobs.at(job_id).get()); // 根据 job 对象，构造了 job_builder 对象
    for (const OperatorConf& op_conf : jobs.at(job_id)->net().op()) {
      if (match.find(op_conf.op_type_case()) == match.end()) { continue; } // 若当前 job 的 op_conf.op_type_case() 不在 OperatorConf::OpTypeCase 匹配集合 match 里，则跳过进入下一个 job 循环
      ParallelBlobConf parallel_blob_conf; 
      GetMemSharingOpBlobInfo(job_builder, op_conf.name(), &parallel_blob_conf); // 获取当前 job 中需要内存复用的 OpBlob 的信息
      auto iter = op_name2parallel_blob_conf->find(op_conf.name());
      if (iter == op_name2parallel_blob_conf->end()) { // 判断 op_name2parallel_blob_conf 是否存在当前 job 的 (op_conf，parallel_blob_conf）信息
        CHECK(op_name2parallel_blob_conf->emplace(op_conf.name(), parallel_blob_conf).second); // 如果没有，则 op_name2parallel_blob_conf 添加当前 job 的信息
      } else {
        CHECK(parallel_blob_conf == iter->second); // 如果有，则校验两者的 parallel_blob_conf 是否相等
      }
    }
  }
}
```

# 系统级 Job

1. user job： 即用户定义的 job，通常为训练或者预测任务；
2. push/pull job： 则是在用户的 user job 编译为可执行 plan 时，系统自动添加的用于处理输入输出的系统级 job；
3. model io job：则是用于初始化/保存/加载模型的系统级 job。

代码功能——编译期数据流转

从数据层面看一下 User Job 的运行过程：首先，User Job 可能有多个输入、多个输出，oneflow 会遍历所有 User Job 中的 Input Op 和 Return Op，针对每个 Input Op，分别构建一个对应的 Push Job；针对每个 Return Op，分别构建一个对应的 Pull Job。

https://pic4.zhimg.com/v2-562d89a68926fd1c3905d9270d34f917_r.jpg

系统自动添加的 Push Job 用于接收输入数据，其 ForeignInput Op 内部维护一个 buffer，该 buffer 等待 Python 端喂数据；Push Job 处理完输入数据 X1 后，由于 X1 在 Push Job 和 User Job 间是内存共享的，可以直接被 User Job 所消费，从而继续被 Op_a、Op_b 处理，最后得到输出数据 Y1；同样，系统添加的 Pull Job 专门用于处理输出数据，Pull Job 中有一个 ForeignOutput Op，其内部同样维护一个 buffer，当往该 buffer 内填完数据以后，python 端对应的 of blob 对象中的 numpy 就拷贝了对应的数据。从而完整整个从输入到输出的数据流转过程。


# MakeModelIoJobs && MakeModelIoV2Jobs<br>

MakeModelInitJob 干了啥？<br>
JobBuilder 类对象通过 `job_builder.AddOps(Device 信息，即 parallel_conf , {xxx_op_conf 的 vector})` 将 op 信息添加到<br>
JobBuilder::job_ 、 JobBuilder::op_name2op_conf_ 、  PlacementGroup 里。<br>

1. JobBuilder 对象添加 foreign_input_op、tick_op 的信息<br>
 `job_builder.AddOps(master_parallel_conf, {foreign_input_op_conf, tick_op_conf}); // 参数包括 Device、foreign_input、tick 信息`<br>

2. JobBuilder 对象添加 var_op 的信息<br>
 `job_builder.AddOps(parallel_blob_conf.parallel_conf(), {new_var_op_conf})`<br>

3. job_builder 添加 model_init_op 的信息<br>
 `job_builder.AddOps(pair.first, {model_init_op_conf}); // job_builder 添加 model_init_op 的信息（parallel_conf，model_init_op_conf）`<br>

model_io_v2_job.cpp<br> 
```.cpp
// MakeModelInitJob("System-ModelInit", &model_init_job, var_op_name2op_conf, var_op_name2parallel_blob_conf);
void MakeModelInitJob(
  const std::string& job_name, Job* job,
  const HashMap<std::string, OperatorConf>& var_op_name2op_conf,
  const HashMap<std::string, ParallelBlobConf>& var_op_name2parallel_blob_conf) {
auto* flag_name2flag_value = job->mutable_job_conf()->mutable_flag_name2flag_value();
(*flag_name2flag_value)["__is_user_function__"].set_at_bool(false);

// 设置当前 job 的 job_conf 的 job_name_ 和 job_type_.predict_conf_
SetModelIoDefaultJobConf(job->mutable_job_conf(), job_name);
Global<InterUserJobInfo>::Get()->set_global_model_init_job_name(job_name);

// 构造当前 job 的 job_builder 对象
JobBuilder job_builder(job);

// 设置 device，ParallelConf::set_device_tag("cpu"), ParallelConf::add_device_name("0:0")
const ParallelConf master_parallel_conf = GenParallelConfOfCpuZeroOnMaster();
const OperatorConf tick_op_conf = GenTickOpConf("System-ModelInit-Tick"); // tick 信号
/**
job 编译到 plan 中，最后会创建一个运行时的 new Runtime()，不过，Runtime 并没有立即执行，而是会等待一个tick信号触发。
通常，系统自动添加的 Push Job 中的 ForeignInput Op 内部维护一个buffer，该 buffer 等待 Python 端喂数据，一旦有数据输入此 op，
将触发 tick 信号，开启整个 Plan 的运行过程。
*/
const OperatorConf foreign_input_op_conf = GenForeignInputOpConf(job_name, 1);

// 将待添加的 op 的 op_conf 添加到 job_ 里，同时将新添加的（op_conf.name(), mut_op_conf）写入 op_name2op_conf_ 中，
// 将 (op_names, parallel_conf) 添加到 parallel_conf2placement_group_ 和 op_name2parallel_conf_ 里
job_builder.AddOps(master_parallel_conf, {foreign_input_op_conf, tick_op_conf}); // 参数包括 Device、foreign_input、tick 信息

if (var_op_name2op_conf.empty()) { return; }
HashMap<ParallelConf, std::vector<OperatorConf>> parallel_conf2variable_op_conf; // Device 上的 Op
for (const auto& pair : var_op_name2op_conf) { // var_op_name2op_conf
  const auto& var_op_name = pair.first;
  const OperatorConf& variable_op_conf = pair.second; // variable_op_conf
  const ParallelBlobConf& parallel_blob_conf = var_op_name2parallel_blob_conf.at(var_op_name);
  parallel_conf2variable_op_conf[parallel_blob_conf.parallel_conf()].push_back(variable_op_conf);
  OperatorConf new_var_op_conf = CloneVariableOpConf(variable_op_conf);
  job_builder.AddOps(parallel_blob_conf.parallel_conf(), {new_var_op_conf});  // job_builder 添加 var_op 的信息（parallel_conf，var_op_conf）
}

// 构造 model_init_op_conf，并用 variable_op_confs 初始化
for (auto& pair : parallel_conf2variable_op_conf) { // parallel_conf2variable_op_conf
  std::vector<OperatorConf>& variable_op_confs = pair.second;
  OperatorConf model_init_op_conf{}; // 构造 model_init_op_conf，并用 variable_op_confs 初始化
  model_init_op_conf.set_name("System-ModelInit-" + NewUniqueId());
  // 获取 model_init_op_conf，并为其预留内存空间
  ModelInitV2OpConf* model_init_conf = model_init_op_conf.mutable_model_init_v2_conf();
  const int64_t num_var = variable_op_confs.size();
  model_init_conf->mutable_ref()->Reserve(num_var);
  model_init_conf->mutable_variable_op_name()->Reserve(num_var); 
  model_init_conf->mutable_original_variable_conf()->Reserve(num_var);

  // 用 variable_op_confs 的信息初始化 model_init_conf 的内容
  for (int64_t i = 0; i < num_var; ++i) {
    model_init_conf->add_ref(GetVariableLbn(variable_op_confs.at(i))); // GenLogicalBlobName
    model_init_conf->add_variable_op_name(variable_op_confs.at(i).name());
    *model_init_conf->add_original_variable_conf() =
        std::move(*variable_op_confs.at(i).mutable_variable_conf());
  }
  job_builder.AddOps(pair.first, {model_init_op_conf}); // job_builder 添加 model_init_op 的信息（parallel_conf，model_init_op_conf）
}
}
```


job_builder.AddOps(parallel_conf 信息, {xxx_op_conf 的 vector}) 干了啥？<br>
将新添加 op 的 op_conf、op_conf、parallel_conf 信息更新（赋值）到 JobBuilder 类对象的成员变量中。<br>

1. JobBuilder::job_: 用新添加 Op 的 op_conf 初始化 JobBuilder::job_ 的 OperatorConf<br>
2. JobBuilder::op_name2op_conf_: JobBuilder::op_name2op_conf_ 添加新加入 Op 的（op_conf.name(), mut_op_conf)，同时 op_names 添加 op_conf.name()<br>
3. PlacementGroup: 将 (op_names, parallel_conf) 添加到 parallel_conf2placement_group_ 和 op_name2parallel_conf_ 里<br>

job_builder.cpp -> JobBuilder::AddOps<br>
```.cpp
// 将待添加的 op 的 op_conf 添加到 job_ 里，同时将新添加的（op_conf.name(), mut_op_conf）写入 op_name2op_conf_ 中，
// 将 (op_names, parallel_conf) 添加到 parallel_conf2placement_group_ 和 op_name2parallel_conf_ 里
void JobBuilder::AddOps(const ParallelConf& parallel_conf,
                      const std::vector<OperatorConf>& op_confs) {
// 做 3 件事情：
// 1. JobBuilder::job_: 用新添加 Op 的 op_conf 初始化 JobBuilder::job_ 的 OperatorConf
// 2. JobBuilder::op_name2op_conf_: JobBuilder::op_name2op_conf_ 添加新加入 Op 的（op_conf.name(), mut_op_conf)，同时 op_names 添加 op_conf.name()
// 3. PlacementGroup: 将 (op_names, parallel_conf) 添加到 parallel_conf2placement_group_ 和 op_name2parallel_conf_ 里

if (op_confs.empty()) { return; }
std::vector<std::string> op_names;
op_names.reserve(op_confs.size());


for (const auto& op_conf : op_confs) {
  CHECK(op_name2op_conf_.find(op_conf.name()) == op_name2op_conf_.end());

  // 将待添加 op 的 op_conf 赋值给 job_ 的 op_conf
  OperatorConf* mut_op_conf = job_->mutable_net()->add_op();
  *mut_op_conf = op_conf;

  // 将新添加的（op_conf.name(), op_conf）写入 op_name2op_conf_ 中
  CHECK(op_name2op_conf_.emplace(op_conf.name(), mut_op_conf).second);
  op_names.emplace_back(op_conf.name());
}

// 将 (op_names, parallel_conf) 添加到 parallel_conf2placement_group_ 和 op_name2parallel_conf_ 里
AddOpNamesToPlacementGroup(op_names, parallel_conf); 
}
```

model_io_v2_job.cpp -> MakeModelIoV2Jobs<br>
```.cpp
// MakeModelIoV2Jobs(jobs, var_op_name2parallel_blob_conf, AppendJob);
void MakeModelIoV2Jobs(const std::vector<std::shared_ptr<Job>>& jobs,
                     const HashMap<std::string, ParallelBlobConf>& var_op_name2parallel_blob_conf,
                     const std::function<void(Job*)>& Handler) {
HashMap<std::string, OperatorConf> var_op_name2op_conf;
FilterVariableOps(jobs, &var_op_name2op_conf);
{
  Job model_init_job;
  MakeModelInitJob("System-ModelInit", &model_init_job, var_op_name2op_conf,
                   var_op_name2parallel_blob_conf);
  Handler(&model_init_job); // 创建 job 对象，将当前 job 添加至 jobs 容器中  
}
{
  Job model_load_job;
  MakeModelLoadJob("System-ModelLoad", &model_load_job, var_op_name2op_conf,
                   var_op_name2parallel_blob_conf);
  Handler(&model_load_job);
}
{
  Job model_save_job;
  MakeModelSaveJob("System-ModelSave", &model_save_job, var_op_name2op_conf,
                   var_op_name2parallel_blob_conf);
  Handler(&model_save_job);
}
}

```

FilterVariableOps 干啥的？<br>
1. 因为 Model Init/Load/Save 都是只针对 variable_op，所以需要将 variable_op 从当前 jobs 的 op 中过滤出来。<br>
2. 过滤掉 jobs 中其它 op，只保留 variable_op。逐个校验 jobs 中当前 job 里 var_op 与 var_op_name2op_conf 的 variable_conf 是否相等。<br>
  
model_io_v2_job.cpp -> FilterVariableOps<br>
```.cpp
// 因为 Model Init/Load/Save 都是只针对 variable_op，所以需要将 variable_op 从当前 jobs 的 op 中过滤出来。
// 过滤掉 jobs 中其它 op，只保留 variable_op。逐个校验 jobs 中当前 job 里 var_op 与 var_op_name2op_conf 的 variable_conf 是否相等。
void FilterVariableOps(const std::vector<std::shared_ptr<Job>>& jobs,
                     HashMap<std::string, OperatorConf>* var_op_name2op_conf) {
FOR_RANGE(int64_t, job_id, 0, jobs.size()) {
  for (const OperatorConf& op_conf : jobs.at(job_id)->net().op()) {
    if (op_conf.has_variable_conf()) { // 过滤掉 jobs 中其它 op，只保留 variable_op
      if (var_op_name2op_conf->find(op_conf.name()) == var_op_name2op_conf->end()) {
        CHECK(var_op_name2op_conf->emplace(op_conf.name(), op_conf).second);
      } else {
        CHECK(CompareVariableOpConf(var_op_name2op_conf->at(op_conf.name()).variable_conf(),
                                    op_conf.variable_conf())); // 逐个校验 jobs 中当前 job 里 var_op 与 var_op_name2op_conf 的 variable_conf 是否相等
      }
    }
  }
}
}
```


# MakePushJob
PushJob:
oneflow 遍历所有 User Job 中的 Input Op，针对每个 Input Op，分别构建一个对应的 Push Job。
系统自动添加的 Push Job 用于接收输入数据，其 `ForeignInput Op` 内部维护一个buffer，该 buffer 等待 Python 端喂数据.

MakePushJob 主要做三件事：
1. job_builder.AddOps 添加 foreign_input_op 的信息；
2. job_builder 添加 output_op 的信息；
3. 最后配置 job 的 name、predict、data_typ 信息。

oneflow.cpp -> MakePushJob

```.cpp
/**
PushJob:
oneflow 遍历所有 User Job 中的 Input Op，针对每个 Input Op，分别构建一个对应的 Push Job。
系统自动添加的 Push Job 用于接收输入数据，其 ForeignInput Op 内部维护一个buffer，该 buffer 等待 Python 端喂数据.

MakePushJob：
1. job_builder.AddOps 添加 foreign_input_op 的信息；
2. job_builder 添加 output_op 的信息；
3. 最后配置 job 的 name、predict、data_typ 信息。
*/

// 外部调用：MakePushJob(std::string("System-Push-") + push_op_name, push_op_name, parallel_blob_conf, push_job)
void MakePushJob(const std::string& job_name, const std::string& op_name,
                 const ParallelBlobConf& parallel_blob_conf, Job* job) {
  auto* flag_name2flag_value = job->mutable_job_conf()->mutable_flag_name2flag_value();
  (*flag_name2flag_value)["__is_user_function__"].set_at_bool(false); // 标识当前 job 是系统级 job
  auto* op_name2job_name =
      Global<InterUserJobInfo>::Get()->mutable_input_or_var_op_name2push_job_name(); // 获取 User Job 中 op_name2job_name 
  CHECK(op_name2job_name->find(op_name) == op_name2job_name->end());
  (*op_name2job_name)[op_name] = job_name; // job_name 赋值给 op_name2job_name
  DataType data_type;
  JobBuilder job_builder(job); // 创建 JobBuilder 对象，将输入 job 的信息添加到 JobBuilder 的成员变量中。 

  // 数据流：Data(Python 端) -> buffer(系统级 ForeignInput Op) 
  OperatorConf foreign_input_op_conf; // ForeignInput Op 内部维护一个 buffer，该 buffer 等待 Python 端喂数据
  {
    // 1. 初始化 blob_conf(边信息): 设置 foreign_input_op_conf （点信息）的 name、out、buffer_name，并利用 op_conf（边）初始化 blob_conf（点）
    // 2. 配置 parallel_conf 的 device 信息
    // 3. 最后 job_builder.AddOps 添加 foreign_input_op 
    foreign_input_op_conf.set_name(std::string("System-Push-ForeignInput_") + NewUniqueId());
    auto* foreign_input_conf = foreign_input_op_conf.mutable_foreign_input_conf();
    foreign_input_conf->set_out("out");
    foreign_input_conf->set_ofblob_buffer_name(GetForeignInputBufferName(job_name));
    auto* blob_conf = foreign_input_conf->mutable_blob_conf();
    InterfaceOpUtil::InitBlobConf(blob_conf, parallel_blob_conf); // 利用 foreign_input_conf 初始化 blob_conf
    data_type = blob_conf->data_type();
    ParallelConf parallel_conf;
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name("0:0");
    job_builder.AddOps(parallel_conf, {foreign_input_op_conf}); // job_builder 添加 foreign_input_op 的信息 （parallel_conf，foreign_input_op_conf）
  }

  // 数据流：buffer(系统级 ForeignInput Op) -> GPU(系统级 Output Op)
  OperatorConf output_op_conf;
  {
    output_op_conf.set_name(op_name);
    auto* output_conf = output_op_conf.mutable_output_conf();
    output_conf->set_in(foreign_input_op_conf.name() + "/out");
    output_conf->set_out("out");
    InterfaceOpUtil::InitBlobConf(output_conf->mutable_blob_conf(), parallel_blob_conf);
    job_builder.AddOps(parallel_blob_conf.parallel_conf(), {output_op_conf});
  } // job_builder 添加 output_op 的信息


  // 配置 job 的 name、predict、data_typ 信息，将 PushJob 的结果保存到 job 中
  auto* job_conf = job->mutable_job_conf();
  job_conf->set_job_name(job_name);
  job_conf->mutable_predict_conf();
  job_conf->set_default_data_type(data_type);
}

```

# MakePullJob
系统添加的 Pull Job 专门用于处理输出数据，Pull Job 中有一个 `ForeignOutput Op`，其内部同样维护一个 buffer，当往该 buffer 内填完数据以后，python 端对应的 of blob 对象中的 numpy 就拷贝了对应的数据。从而完整整个从输入到输出的数据流转过程。

oneflow.cpp -> MakePullJob
```.cpp
void MakePullJob(const std::string& job_name, const std::string& op_name,
                 const ParallelBlobConf& parallel_blob_conf, Job* job) {
  auto* flag_name2flag_value = job->mutable_job_conf()->mutable_flag_name2flag_value();
  (*flag_name2flag_value)["__is_user_function__"].set_at_bool(false);
  auto* op_name2job_name =
      Global<InterUserJobInfo>::Get()->mutable_output_or_var_op_name2pull_job_name();
  CHECK(op_name2job_name->find(op_name) == op_name2job_name->end());
  (*op_name2job_name)[op_name] = job_name;
  DataType data_type;
  JobBuilder job_builder(job);
  OperatorConf input_op_conf;
  {
    input_op_conf.set_name(op_name);
    auto* input_conf = input_op_conf.mutable_input_conf();
    input_conf->set_out("out");
    auto* blob_conf = input_conf->mutable_blob_conf();
    InterfaceOpUtil::InitBlobConf(blob_conf, parallel_blob_conf);
    data_type = blob_conf->data_type();
    job_builder.AddOps(parallel_blob_conf.parallel_conf(), {input_op_conf});
  }
  OperatorConf foreign_output_op_conf;
  {
    foreign_output_op_conf.set_name(std::string("System-Pull-ForeignOutput_") + NewUniqueId());
    auto* foreign_output_conf = foreign_output_op_conf.mutable_foreign_output_conf();
    foreign_output_conf->set_in(input_op_conf.name() + "/out");
    foreign_output_conf->set_ofblob_buffer_name(GetForeignOutputBufferName(job_name));
    ParallelConf parallel_conf;
    parallel_conf.set_device_tag("cpu");
    parallel_conf.add_device_name("0:0");
    job_builder.AddOps(parallel_conf, {foreign_output_op_conf});
  }
  auto* job_conf = job->mutable_job_conf();
  job_conf->set_job_name(job_name);
  job_conf->mutable_predict_conf();
  job_conf->set_default_data_type(data_type);
}
```

# CompileCurJobOnMaster

# compiler.cpp -> Compiler::Compile

# job_completer.cpp -> JobCompleter::Complete(Job* job)


# CompileJobsAndMergePlans

