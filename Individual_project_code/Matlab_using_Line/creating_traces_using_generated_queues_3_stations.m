%% define the queue first

clear all

for i=1:100
    
    x_range = 50:100;
    s_range = 4:100;
    mu_range = 4:40;
    
    s_1 = randsample(s_range,1);
    s_2 = randsample(s_range,1);
    s_3 = randsample(s_range,1);
    
    mu_1 = randsample(mu_range,1);
    mu_2 = randsample(mu_range,1);
    mu_3 = randsample(mu_range,1);
    
    x_1 = randsample(x_range, 1);
    x_2 = randsample(x_range, 1);
    x_3 = randsample(x_range, 1);
    
    x = [x_1;x_2;x_3];
    s = [200,50,10];
    mu = [10,20,40];
    
    model = Network('myModel');

    %% Block 1: nodes
    node{1} = Queue(model, 'QueueStation1', SchedStrategy.FCFS);
    node{2} = Queue(model, 'QueueStation2', SchedStrategy.FCFS);
    node{3} = Queue(model, 'QueueStation3', SchedStrategy.FCFS);
    % queue4 = Queue(model, 'QueueStation4', SchedStrategy.FCFS);
    % queue5 = Queue(model, 'QueueStation5', SchedStrategy.FCFS);

    %% Set number of servers for each queueing station node
    node{1}.setNumberOfServers(s(1));
    node{2}.setNumberOfServers(s(2));
    node{3}.setNumberOfServers(s(3));
    % queue4.setNumberOfServers(s(4));
    % queue5.setNumberOfServers(s(5));

    %% Block 2: Classes

    cclass1 = ClosedClass(model, 'class1', x(1), node{1});
    % cclass2 = ClosedClass(model, 'class2', x(2), queue2);
    % cclass3 = ClosedClass(model, 'class3', x(3), queue3);

    node{2}.setState(x(2))
    node{3}.setState(x(3))

    %% Might have to set service rates like this and create loads of them so let's see

    %% Set service rates for each queueing station node 

    % First class starting at node 1

    node{1}.setService(cclass1, Exp(mu(1)));
    node{2}.setService(cclass1, Exp(mu(2)));
    node{3}.setService(cclass1, Exp(mu(3)));

    % Second class starting at node 2

    % queue1.setService(cclass2, Exp(mu(1)));
    % queue2.setService(cclass2, Exp(mu(2)));
    % queue3.setService(cclass2, Exp(mu(3)));
    % 
    % % 3rd class starting at node 3
    % 
    % queue1.setService(cclass3, Exp(mu(1)));
    % queue2.setService(cclass3, Exp(mu(2)));
    % queue3.setService(cclass3, Exp(mu(3)));

    %% Block 3: Topology using Routing Matrix P

    T = [0,0.5,0.5;0.5,0,0.5;0.5,0.5,0];

    P = model.initRoutingMatrix;
    P{cclass1} = T;
    % P{cclass2} = T;
    % P{cclass3} = T;
    model.link(P);

    %% Metrics

    solver = SolverJMT(model);

    [QN,UN,RN,TN] = solver.getAvg();

    [Qt,Ut,Tt] = model.getTranHandles();

    [QNt,UNt,TNt] = SolverJMT(model,'force', true, 'timespan',[0,10,0.01]).getTranAvg(Qt,Ut,Tt);

    plot(QNt{1,1}.t, QNt{1,1}.metric);
    plot(QNt{2,1}.t, QNt{2,1}.metric);
    plot(QNt{3,1}.t, QNt{3,1}.metric);

    time_intervals = QNt{1,1}.t;
    av_q_len_station1 = QNt{1,1}.metric;
    av_q_len_station2 = QNt{2,1}.metric;
    av_q_len_station3 = QNt{3,1}.metric;
    
    total_size_1 = size(av_q_len_station1);
    total_size_2 = size(av_q_len_station2);
    total_size_3 = size(av_q_len_station3);
    time_size = size(time_intervals);
    
    total_length_1 = total_size_1(1);
    total_length_2 = total_size_2(1);
    total_length_3 = total_size_3(1);
    time_length = time_size(1);
    
    av_q_len_station1 = av_q_len_station1(1:100:total_length_1);
    av_q_len_station2 = av_q_len_station2(1:100:total_length_2);
    av_q_len_station3 = av_q_len_station3(1:100:total_length_3);
    time_intervals = time_intervals(1:100:time_length);
    
    test_size = size(av_q_len_station1)

    average_queue_length_trace = [time_intervals, av_q_len_station1, av_q_len_station2, av_q_len_station3];

    % save('average_queue_length_trace.mat','trace_1');

    save(['average_queue_length_trace_' num2str(i) '.mat'],'average_queue_length_trace')
    
end 

