import simpy
import os
import random
import numpy as np
import math
from collections import OrderedDict
import functools


class Part(object):
    def __init__(self, name, type, process_time, weight, K, process_list):
        # 해당 Part 번호
        self.id = name
        # job type
        self.type = type
        # p_j에 해당 (job type j의 average process time)
        self.process_time = {'process_1': np.average(process_time['process_1'], axis=1)[self.type]}
        # determined process time
        self.real_proc_time = None
        # list of process that this part go through
        self.process_list = process_list
        # p_ij에 해당
        self.p_ij = {'process_1': process_time['process_1'][self.type][:]}
        # p_ij 중 max
        self.max_process_time = {'process_1': np.max(np.array(self.p_ij['process_1']))}
        # p_ij 중 min
        self.min_process_time = {'process_1': np.min(np.array(self.p_ij['process_1']))}
        # 작업을 완료한 공정의 수
        self.step = 0
        # part 의 due date
        self.due_date = {}
        # due date generating factor
        self.K = K
        self.W_ij = None
        # job type j 의 weight
        self.weight = weight
        self.completion_time = {'process_1': None}

    def set_due_date(self, arrival_time):
        for process in self.process_list[:-1]:
            self.due_date[process] = arrival_time + self.K[process] * self.process_time[process]
            arrival_time += self.process_time[process]


class Source(object):
    def __init__(self, env, IAT, weight, job_type, job_type_list, process_time, model, process_list, machine_num, part_num, K):
        self.env = env
        self.name = 'Source'
        self.parts_sent = 0
        self.IAT = IAT    # part 의 iat list (매 episode 마다 바뀜)
        self.weight = weight    # w_j
        self.job_type = job_type
        self.job_type_list = job_type_list  # 생성되는 part 의 job type list (매 episode 마다 바뀜)
        self.process_time = process_time    # p_ij

        env.process(self.job_generating_process(self.IAT, self.job_type_list))

        # 각 job type 별 생성된 job 수
        self.generated_job_types = np.zeros(len(self.job_type))
        self.model = model
        self.process_list = process_list
        self.machine_num = machine_num
        self.part_num = part_num
        self.K = K

    def job_generating_process(self, IAT, job_type_list):
        while True:
            for jb_type, iat in zip(job_type_list, IAT):
                yield self.env.timeout(iat)
                self.generated_job_types[jb_type] += 1
                w = self.weight[jb_type]
                p = self.process_time

                # generate job
                part = Part(name='job{0}_{1}'.format(jb_type, self.generated_job_types[jb_type]), type=jb_type,
                            process_time=p, weight=w, K=self.K, process_list=self.process_list)
                # set due date
                part.set_due_date(self.env.now)

                # put job batch to next process buffer_to_machine
                self.model[self.process_list[part.step]].buffer_to_machine.put(part)
                # 새로운 part arrival 발생할 때마다 해당 part 를 기록하는 list
                self.model[self.process_list[part.step]].new_arrivals.append(part)
                self.parts_sent += 1
            if self.parts_sent == self.part_num:
                break


class Process(object):
    def __init__(self, env, name, machine_num, model, process_list, capacity=float('inf'),
                 capa_to_machine=float('inf'), capa_to_process=float('inf')):
        self.env = env
        self.name = name    # process name
        self.model = model
        self.machine_num = machine_num
        self.process_list = process_list    # part 들이 통과하는 process list
        self.routing_ongoing = False    # routing 이 진행되고 있는지 여부
        self.parts_sent = 0
        self.parts_routed = 0
        self.new_arrivals = []
        self.new_idles = []
        self.buffer_to_machine = simpy.FilterStore(env, capacity=capa_to_machine)   # part queue (대기하는 part 들이 존재)
        self.machine_store = simpy.FilterStore(env, capacity=machine_num)   # machine store (가용한 machine 들이 들어있음)
        self.machines = [Machine(env, self.name, 'Machine_{0}'.format(i), idx=i,
                                 model=model, process_list=self.process_list) for i in range(machine_num)]
        # 처음에 machine 객체들을 machine store 에 넣어 둔다
        for i in range(self.machine_store.capacity):
            self.machine_store.put(self.machines[i])

        env.process(self._to_machine())
        env.process(self.check_idle_machine())

        # idle machine 이 있을 때 열리고(succeed) 없을 때 닫혀있는 스위치(valve?) event
        # check_idle_machine(Process)에 의해서 제어 됨
        self.idle_machine = env.event()
        # idle machine 을 check 할 시점이 되면 열리고(succeed) 없을 때 닫혀있는 스위치(valve?) event
        self.wait_before_check = env.event()
        self.action = 0
        self.working_process_list = dict()  # Process 의 현재 작업 진행중인 machines

    def _to_machine(self):
        while True:
            self.routing_logic = None

            # If there exist idle machines and also parts in buffer_to_machine
            # Then take action (until one of the items becomes zero)
            if len(self.buffer_to_machine.items) != 0 and len(self.machine_store.items) != 0:
                while len(self.buffer_to_machine.items) != 0 and len(self.machine_store.items) != 0:
                    self.routing_logic = self.routing(self.action)
                    self.routing_ongoing = True

                    part = yield self.buffer_to_machine.get()
                    machine = yield self.machine_store.get()

                    self.parts_routed += 1
                    machine.part_in_machine.append(part)
                    self.working_process_list[machine.name] = self.env.process(machine.work(part, self.machine_store))
                self.routing_ongoing = False

            # If there exist idle machine, but no part in buffer_to_machine
            # Then wait until new part arrival
            elif len(self.buffer_to_machine.items) == 0 and len(self.machine_store.items) != 0:
                part = yield self.buffer_to_machine.get()
                # dispatching priority 를 계산하기 위한 잠깐 part 를 buffer_to_machine 에 넣어줌
                self.buffer_to_machine.put(part)
                self.routing_logic = self.routing(self.action)
                self.routing_ongoing = True
                part = yield self.buffer_to_machine.get()
                machine = yield self.machine_store.get()

                self.parts_routed += 1
                machine.part_in_machine.append(part)
                self.working_process_list[machine.name] = self.env.process(machine.work(part, self.machine_store))
                self.routing_ongoing = False

            # wait until there exists an idle machine
            else:
                yield self.idle_machine     # idle machine 있을 때만 열리는 valve

    def check_idle_machine(self):  # idle_machine event(valve)를 제어하는 Process
        while True:
            if len(self.machine_store.items) != 0:
                # idle machine 있을 시 idle_machine event(valve)를 열었다가 바로 닫는다.
                self.idle_machine.succeed()
                self.idle_machine = self.env.event()

            yield self.wait_before_check  # wait for time to check(valve)
            # machine 이 machine_store 에 반납된 후마다 열렸다가 바로 닫힘

    # i : idle machine index , j : part in buffer_to_machine index
    # idle_machines (list) 와 parts_in_buffer (list) 이용해서 indexing
    # idle machine name 과 part in buffer_to_machine id
    def routing(self, action):
        if action == 0:  # WSPT
            if len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) != 0:
                # machine_store.items sorting
                for i in self.machine_store.items:
                    W_i = list()
                    for j in self.buffer_to_machine.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = p_ij / j.weight
                        W_i.append(w_ij)
                    i.W_ij = min(W_i)
                self.machine_store.items.sort(key=lambda machine: machine.W_ij)

                # buffer_to_machine.items sorting
                for j in self.buffer_to_machine.items:
                    W_j = list()
                    for i in self.machine_store.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = p_ij / j.weight
                        W_j.append(w_ij)
                    j.W_ij = min(W_j)
                self.buffer_to_machine.items.sort(key=lambda part: part.W_ij)
            return 0

        elif action == 1:  # WMDD
            if len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) != 0:
                # machine_store.items sorting
                for i in self.machine_store.items:
                    W_i = list()
                    for j in self.buffer_to_machine.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = max(p_ij, j.due_date[self.process_list[j.step]] - self.env.now)
                        w_ij = w_ij / j.weight
                        W_i.append(w_ij)
                    i.W_ij = min(W_i)
                self.machine_store.items.sort(key=lambda machine: machine.W_ij)

                # buffer_to_machine.items sorting
                for j in self.buffer_to_machine.items:
                    W_j = list()
                    for i in self.machine_store.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = max(p_ij, j.due_date[self.process_list[j.step]] - self.env.now)
                        w_ij = w_ij / j.weight
                        W_j.append(w_ij)
                    j.W_ij = min(W_j)
                self.buffer_to_machine.items.sort(key=lambda part: part.W_ij)
            return 1

        elif action == 2:  # ATC
            if len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) != 0:
                # machine_store.items sorting
                p = 0.0  # average nominal processing time
                for part in self.buffer_to_machine.items:
                    p += part.process_time[self.process_list[part.step]]
                p = p / len(self.buffer_to_machine.items)

                h = 2.3  # look-ahead parameter

                for i in self.machine_store.items:
                    W_i = list()
                    for j in self.buffer_to_machine.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = -1 * max(0, j.due_date[self.process_list[j.step]] - self.env.now - p_ij) / (h * p)
                        w_ij = j.weight / p_ij * math.exp(w_ij)
                        W_i.append(w_ij)
                    i.W_ij = max(W_i)
                self.machine_store.items.sort(key=lambda machine: machine.W_ij, reverse=True)

                # buffer_to_machine.items sorting
                for j in self.buffer_to_machine.items:
                    W_j = list()
                    for i in self.machine_store.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = -1 * max(0, j.due_date[self.process_list[j.step]] - self.env.now - p_ij) / (h * p)
                        w_ij = j.weight / p_ij * math.exp(w_ij)
                        W_j.append(w_ij)
                    j.W_ij = max(W_j)
                self.buffer_to_machine.items.sort(key=lambda part: part.W_ij, reverse=True)
            return 2
        elif action == 3:  # WCOVERT
            if len(self.machine_store.items) != 0 and len(self.buffer_to_machine.items) != 0:
                # machine_store.items sorting
                K_t = 2.3  # approximation factor

                for i in self.machine_store.items:
                    W_i = list()
                    for j in self.buffer_to_machine.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = 1 - max(0, j.due_date[self.process_list[j.step]] - self.env.now - p_ij) / (K_t * p_ij)
                        w_ij = j.weight / p_ij * max(0, w_ij)
                        W_i.append(w_ij)
                    i.W_ij = max(W_i)
                self.machine_store.items.sort(key=lambda machine: machine.W_ij, reverse=True)

                # buffer_to_machine.items sorting
                for j in self.buffer_to_machine.items:
                    W_j = list()
                    for i in self.machine_store.items:
                        p_ij = j.p_ij[self.process_list[j.step]][i.idx]
                        w_ij = 1 - max(0, j.due_date[self.process_list[j.step]] - self.env.now - p_ij) / (K_t * p_ij)
                        w_ij = j.weight / p_ij * max(0, w_ij)
                        W_j.append(w_ij)
                    j.W_ij = max(W_j)
                self.buffer_to_machine.items.sort(key=lambda part: part.W_ij, reverse=True)
            return 3
        elif action == 4:   # do nothing when there are no new part arrivals or no idle machines
            return 4            # do nothing


class Machine(object):
    def __init__(self, env, process_name, name, idx, model, process_list):
        self.env = env
        self.process_name = process_name    # 해당 machine 이 속해있는 process name
        self.name = name        # machine name
        self.idx = idx          # machine index
        self.model = model
        self.process_list = process_list
        self.part_in_machine = []   # 해당 machine 에서 작업중인 part list
        self.start_work = None
        self.W_ij = None

    def work(self, part, machine_store):
        # process_time
        proc_time = part.p_ij[self.process_name][self.idx]
        # proc_time = np.random.triangular(left=0.9 * proc_time, mode=proc_time, right=1.1 * proc_time)
        part.real_proc_time = proc_time

        self.start_work = self.env.now

        yield self.env.timeout(proc_time)

        _ = self.part_in_machine.pop(0)     # 작업이 완료되면 part_in_machine 을 비워준다
        # return machine object(self) to the machine store
        machine_store.put(self)

        # next process
        next_process_name = self.process_list[part.step + 1]
        next_process = self.model[next_process_name]
        process = self.model[self.process_name]

        if next_process.__class__.__name__ == 'Process':    # if next process is 'Process'
            # part transfer
            part.completion_time[self.process_name] = self.env.now
            next_process.buffer_to_machine.put(part)
            next_process.new_arrivals.append(part)
            process.new_idles.append(part)
        else:       # if next process is 'Sink'
            part.completion_time[self.process_name] = self.env.now
            next_process.put(part)
            process.new_idles.append(part)

        part.step += 1
        self.model[self.process_name].parts_sent += 1

        # Idle machine occurs -> check
        # Idle machine check 시점이 되면 wait_before_check(valve)를 열었다가 바로 닫는다
        self.model[self.process_name].wait_before_check.succeed()
        self.model[self.process_name].wait_before_check = self.env.event()


class Sink(object):
    def __init__(self, env):
        self.env = env
        self.name = 'Sink'
        self.parts_rec = 0
        self.last_arrival = 0.0
        self.sink = list()      # Sink 에 도착한 part 들을 넣어두는 공간

    def put(self, part):
        self.parts_rec += 1
        self.last_arrival = self.env.now
        self.sink.append(part)