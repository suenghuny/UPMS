import numpy as np
from environment.simulation import *

class UPMSP:
    def __init__(self, num_jt=10, num_j=1000, num_m=8, log_dir=None, K=1):
        self.num_jt = num_jt
        self.num_machine = num_m
        self.jobtypes = [i for i in range(num_jt)]  # 1~10
        self.p_ij, self.p_j, self.weight = self._generating_data()
        self.num_job = num_j
        self.log_dir = log_dir
        self.jobtype_assigned = list()  # 어느 job이 어느 jobtype에 할당되는 지
        self.job_list = list()          # 모델링된 Job class를 저장할 리스트

        self.K = K
        self.done = False
        self.tardiness = 0.0
        self.e = 0
        self.time = 0

        self.mapping = {0: "WSPT", 1: "WMDD", 2: "ATC", 3: "WCOVERT"}

        self.sim_env, self.process_dict, self.source_dict, self.sink, self.routing, self.monitor = self._modeling()

    def step(self, action):
        done = False
        self.previous_time_step = self.sim_env.now
        routing_rule = self.mapping[action]

        self.routing.decision.succeed(routing_rule)
        self.routing.indicator = False

        while True:
            if self.routing.indicator:
                if self.sim_env.now != self.time:
                    self.time = self.sim_env.now
                break

            if self.sink.finished_job == self.num_job:
                done = True
                self.sim_env.run()
                if self.e % 50 == 0:
                    self.monitor.save_tracer()
                # self.monitor.save_tracer()
                break

            if len(self.sim_env._queue) == 0:
                self.monitor.save_tracer()
                print(self.monitor.filepath)
            self.sim_env.step()

        reward = self._calculate_reward()
        next_state = self._get_state()

        return next_state, reward, done

    def reset(self):
        self.e = self.e + 1 if self.e > 1 else 1  # episode
        self.p_ij, self.p_j, self.weight = self._generating_data()
        self.sim_env, self.process_dict, self.source_dict, self.sink, self.routing, self.monitor = self._modeling()
        self.done = False
        self.monitor.reset()

        self.tardiness = 0

        while True:
            # Check whether there is any decision time step
            if self.routing.indicator:
                break

            self.sim_env.step()

        return self._get_state()

    def _modeling(self):
        env = simpy.Environment()

        monitor = Monitor(self.log_dir + '/log_%d.csv'% self.e)
        # monitor = Monitor("C:/Users/sohyon/PycharmProjects/UPJSP_SH/environment/result/log_{0}.csv".format(self.e))
        process_dict = dict()
        source_dict = dict()
        jt_dict = dict()  # {"JobType 0" : [Job class(), ... ], ... }
        time_dict = dict()  # {"JobType 0" : [pij,...], ... }
        routing = Routing(env, process_dict, source_dict, monitor, self.weight)

        # 0에서 9까지 랜덤으로 배정
        self.jobtype_assigned = np.random.randint(low=0, high=10, size=self.num_job)
        for i in range(self.num_job):
            jt = self.jobtype_assigned[i]
            if "JobType {0}".format(jt) not in jt_dict.keys():
                jt_dict["JobType {0}".format(jt)] = list()
                time_dict["JobType {0}".format(jt)] = self.p_ij[jt]
            jt_dict["JobType {0}".format(jt)].append(
                Job("Job {0}-{1}".format(jt, i), self.p_ij[jt], job_type=jt))

        sink = Sink(env, monitor, jt_dict, self.num_job, source_dict, self.weight)

        for jt_name in jt_dict.keys():
            source_dict["Source {0}".format(int(jt_name[-1]))] = Source("Source {0}".format(int(jt_name[-1])), env,
                                                                        routing, monitor, jt_dict, self.p_j, self.K,
                                                                        self.num_machine)

        for i in range(self.num_machine):
            process_dict["Machine {0}".format(i)] = Process(env, "Machine {0}".format(i), sink, routing, monitor)

        return env, process_dict, source_dict, sink, routing, monitor

    def _get_state(self):
        # define 8 features
        f_1 = np.zeros(self.num_jt)       # placeholder for feature 1
        f_2 = np.zeros(self.num_machine)  # placeholder for feature 2
        f_3 = np.zeros(self.num_machine)  # placeholder for feature 3
        f_4 = np.zeros(self.num_machine)  # placeholder for feature 4
        f_5 = np.zeros(self.num_jt)       # placeholder for feature 5
        f_6 = np.zeros(self.num_jt)       # placeholder for feature 6
        f_7 = np.zeros(self.num_jt)       # placeholder for feature 7
        f_8 = np.zeros([self.num_jt, 4])  # placeholder for feature 8

        for jt_name in self.source_dict.keys():    # for loop by job type
            j = int(jt_name[-1])                   # j : job type index


            """
            To do 1 : count the number of job type j and fill f_1 by the paper's instruction
            - f_1 (job-related feature) : the number of non-processes job in job type j
            - Given 1 : Utilize Job object(class Job)'s storage object(self.routing.queue.items)(i.e. job objects in self.routing.queue.items) 
            - Given 2 : Utilize Job object(class Job)'s job type attribute(Job.job_type)
            """


        for i in range(self.num_machine):                            # for loop by machine
            machine = self.process_dict["Machine {0}".format(i)]     # machine object


            """
            To do 2 : check if machine is idle and fill f_2 by the paper's instruction
            - f_2 (machine-related feature) : what job type is processed in machine
            - Given 1 : Utilize machine(class Process)'s idle attribute(machine.idle)
            - Given 2 : Utilize machine's current processing job attribute(machine.job) (if machine is idle, machine.job is None)
            """



            """
            To do 3 : calculate remaining processing time and fill f_3 by the paper's insturction
            - f_3 (machine-related feature) : how much the time is remaining
            - Given 1 : Utilize machine object(class Process)'s planned finish time attribute(machine.planned_finish_time)
            - Given 2 : Utilize simulation's current time(self.sim_env.now)
            - Given 3 : Utilize job type j's average(or nominal) processing time(self.p_j)
            - Given 4 : Utilize machine's current processing job attribute(machine.job) (if machine is idle, machine.job is None)
            """

            """
            To do 4 : calculate due remaining and fill f_4 by the paper's insturction
            - f_4 (machine-related feature) : how much the time is remaining to go to the due date
            - Given 1 : Utilize due date of machine object(class Process)'s current working job(job.due_date)
            - Given 2 : Utilize simulation's current time(self.sim_env.now)
            - Given 3 : Utilize job type j's average(or nominal) processing time(self.p_j)
            - Given 4 : Utilize machine's current processing job attribute(machine.job) (if machine is idle, machine.job is None)
            """

        for jt_name in self.source_dict.keys():                                                       # for loop by job type
            j = int(jt_name[-1])
            job_list = [job for job in self.routing.queue.items if job.job_type == j]  # job objects whose job type is j
            jt_duedates = [job.due_date for job in job_list]                           # due date of jobs whose job type is j

            """
            To do 5 : calculate waiting job's minimal tightness of the due date allowance and fill feature 5 
            """

            """
            To do 6 : calculate waiting job's maximal tightness of the due date allowance and fill feature 6
            """

            """
            To do 7 : calculate waiting job's average tightness of the due date allowance and fill feature 7 
            """

            """
            To do 8 : calculate each job type's tightness of the due date allowance and fill f_8 by the paper's insturction(utilize each object's already mentioned attribute) 
            - Given 1 : Utilize job type j's processing time on machine i(self.p_ji)
            - Given 2 : Utilize job_list(defined in line 173) and jt_duedates(defined in line 174)
            """


        f_8 = f_8.flatten()
        state = np.concatenate((f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8), axis=None)
        return state

    def _calculate_reward(self):
        reward = 0
        finished_jobs = copy.deepcopy(self.sink.job_list)
        for job in finished_jobs:
            jt = job.job_type
            w_j = self.weight[jt]

            tardiness = min(job.due_date - job.completion_time, 0)

            reward += w_j * tardiness * (1/1000)

        self.sink.job_list = list()

        return reward

    def _generating_data(self):
        processing_time = [[np.random.uniform(low=1, high=20) for _ in range(self.num_machine)] for _ in range(self.num_jt)]
        p_j = [np.mean(jt_pt) for jt_pt in processing_time]
        weight = list(np.random.uniform(low=0, high=5, size=self.num_jt))

        return processing_time, p_j, weight


