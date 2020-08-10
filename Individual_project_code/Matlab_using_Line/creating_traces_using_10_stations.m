%% define the queue first

clear all

for i=1:100
    
    x_range = 40:100;
    s_range = 4:100;
    mu_range = 4:40;
    
    s_1 = randsample(s_range,1);
    s_2 = randsample(s_range,1);
    s_3 = randsample(s_range,1);
    s_4 = randsample(s_range,1);
    s_5 = randsample(s_range,1);
    s_6 = randsample(s_range,1);
    s_7 = randsample(s_range,1);
    s_8 = randsample(s_range,1);
    s_9 = randsample(s_range,1);
    s_10 = randsample(s_range,1);
    
    mu_1 = randsample(mu_range,1);
    mu_2 = randsample(mu_range,1);
    mu_3 = randsample(mu_range,1);
    mu_4 = randsample(mu_range,1);
    mu_5 = randsample(mu_range,1);
    mu_6 = randsample(mu_range,1);
    mu_7 = randsample(mu_range,1);
    mu_8 = randsample(mu_range,1);
    mu_9 = randsample(mu_range,1);
    mu_10 = randsample(mu_range,1);
    
    x_1 = randsample(x_range, 1);
    x_2 = randsample(x_range, 1);
    x_3 = randsample(x_range, 1);
    x_4 = randsample(x_range, 1);
    x_5 = randsample(x_range, 1);
    x_6 = randsample(x_range, 1);
    x_7 = randsample(x_range, 1);
    x_8 = randsample(x_range, 1);
    x_9 = randsample(x_range, 1);
    x_10 = randsample(x_range, 1);
    
    x = [x_1;x_2;x_3;x_4;x_5;x_6;x_7;x_8;x_9;x_10];
    s = [80,20,30,20,40,20,50,20,80,30];
    mu = [200,100,50,100,200,500,200,100,200,100];
    
    model = Network('myModel');

    %% Block 1: nodes
    node{1} = Queue(model, 'QueueStation1', SchedStrategy.FCFS);
    node{2} = Queue(model, 'QueueStation2', SchedStrategy.FCFS);
    node{3} = Queue(model, 'QueueStation3', SchedStrategy.FCFS);
    node{4} = Queue(model, 'QueueStation4', SchedStrategy.FCFS);
    node{5} = Queue(model, 'QueueStation5', SchedStrategy.FCFS);
    node{6} = Queue(model, 'QueueStation6', SchedStrategy.FCFS);
    node{7} = Queue(model, 'QueueStation7', SchedStrategy.FCFS);
    node{8} = Queue(model, 'QueueStation8', SchedStrategy.FCFS);
    node{9} = Queue(model, 'QueueStation9', SchedStrategy.FCFS);
    node{10} = Queue(model, 'QueueStation10', SchedStrategy.FCFS);

    %% Set number of servers for each queueing station node
    node{1}.setNumberOfServers(s(1));
    node{2}.setNumberOfServers(s(2));
    node{3}.setNumberOfServers(s(3));
    node{4}.setNumberOfServers(s(4));
    node{5}.setNumberOfServers(s(5));
    node{6}.setNumberOfServers(s(6));
    node{7}.setNumberOfServers(s(7));
    node{8}.setNumberOfServers(s(8));
    node{9}.setNumberOfServers(s(9));
    node{10}.setNumberOfServers(s(10));

    %% Block 2: Classes

    cclass1 = ClosedClass(model, 'class1', x(1), node{1});
    % cclass2 = ClosedClass(model, 'class2', x(2), queue2);
    % cclass3 = ClosedClass(model, 'class3', x(3), queue3);

    node{2}.setState(x(2));
    node{3}.setState(x(3));
    node{4}.setState(x(4));
    node{5}.setState(x(5));
    node{6}.setState(x(6));
    node{7}.setState(x(7));
    node{8}.setState(x(8));
    node{9}.setState(x(9));
    node{10}.setState(x(10));

    %% Might have to set service rates like this and create loads of them so let's see

    %% Set service rates for each queueing station node 

    % First class starting at node 1

    node{1}.setService(cclass1, Exp(mu(1)));
    node{2}.setService(cclass1, Exp(mu(2)));
    node{3}.setService(cclass1, Exp(mu(3)));
    node{4}.setService(cclass1, Exp(mu(4)));
    node{5}.setService(cclass1, Exp(mu(5)));
    node{6}.setService(cclass1, Exp(mu(6)));
    node{7}.setService(cclass1, Exp(mu(7)));
    node{8}.setService(cclass1, Exp(mu(8)));
    node{9}.setService(cclass1, Exp(mu(9)));
    node{10}.setService(cclass1, Exp(mu(10)));


    %% Block 3: Topology using Routing Matrix P

    T = [0,0.1,0.1,0.1,0.3,0,0.1,0.1,0.1,0.1;
        0,0.05,0.15,0.2,0.1,0.2,0.1,0,0.1,0.1;
        0.1,0.1,0,0.1,0.2,0.1,0.1,0.1,0.1,0.1;
        0.2,0.1,0,0.2,0.1,0,0.1,0.1,0.1,0.1;
        0.2,0.1,0.1,0.1,0,0.1,0.1,0.1,0.1,0.1;
        0.1,0.1,0.2,0.1,0.1,0,0.1,0.1,0.1,0.1;
        0.1,0.2,0.1,0.1,0.1,0.1,0,0.1,0.1,0.1;
        0.1,0.1,0.1,0.1,0.1,0,0.1,0.2,0.1,0.1;
        0.1,0.1,0.1,0.1,0.1,0.2,0.1,0.1,0,0.1;
        0.1,0.1,0.1,0.1,0.1,0.2,0.1,0.1,0.1,0];

    P = model.initRoutingMatrix;
    P{cclass1} = T;
    % P{cclass2} = T;
    % P{cclass3} = T;
    model.link(P);

    %% Metrics

    solver = SolverJMT(model);

    [QN,UN,RN,TN] = solver.getAvg();

    [Qt,Ut,Tt] = model.getTranHandles();

    [QNt,UNt,TNt] = SolverJMT(model,'force', true, 'timespan',[0,10]).getTranAvg(Qt,Ut,Tt);

    time_intervals = QNt{1,1}.t;
    av_q_len_station1 = QNt{1,1}.metric;
    av_q_len_station2 = QNt{2,1}.metric;
    av_q_len_station3 = QNt{3,1}.metric;
    av_q_len_station4 = QNt{4,1}.metric;
    av_q_len_station5 = QNt{5,1}.metric;
    av_q_len_station6 = QNt{6,1}.metric;
    av_q_len_station7 = QNt{7,1}.metric;
    av_q_len_station8 = QNt{8,1}.metric;
    av_q_len_station9 = QNt{9,1}.metric;
    av_q_len_station10 = QNt{10,1}.metric;
    
    total_size_1 = size(av_q_len_station1);
    total_size_2 = size(av_q_len_station2);
    total_size_3 = size(av_q_len_station3);
    total_size_4 = size(av_q_len_station4);
    total_size_5 = size(av_q_len_station5);
    total_size_6 = size(av_q_len_station6);
    total_size_7 = size(av_q_len_station7);
    total_size_8 = size(av_q_len_station8);
    total_size_9 = size(av_q_len_station9);
    total_size_10 = size(av_q_len_station10);
    
    time_size = size(time_intervals);
    
    total_length_1 = total_size_1(1);
    total_length_2 = total_size_2(1);
    total_length_3 = total_size_3(1);
    total_length_4 = total_size_4(1);
    total_length_5 = total_size_5(1);
    total_length_6 = total_size_6(1);
    total_length_7 = total_size_7(1);
    total_length_8 = total_size_8(1);
    total_length_9 = total_size_9(1);
    total_length_10 = total_size_10(1);
    time_length = time_size(1);
    
    av_q_len_station1 = av_q_len_station1(1:100:total_length_1);
    av_q_len_station2 = av_q_len_station2(1:100:total_length_2);
    av_q_len_station3 = av_q_len_station3(1:100:total_length_3);
    av_q_len_station4 = av_q_len_station4(1:100:total_length_4);
    av_q_len_station5 = av_q_len_station5(1:100:total_length_5);
    av_q_len_station6 = av_q_len_station6(1:100:total_length_6);
    av_q_len_station7 = av_q_len_station7(1:100:total_length_7);
    av_q_len_station8 = av_q_len_station8(1:100:total_length_8);
    av_q_len_station9 = av_q_len_station9(1:100:total_length_9);
    av_q_len_station10 = av_q_len_station10(1:100:total_length_10);
    time_intervals = time_intervals(1:100:time_length);
    
    test_size = size(av_q_len_station1);

    average_queue_length_trace = [time_intervals, av_q_len_station1, av_q_len_station2, av_q_len_station3, av_q_len_station4, av_q_len_station5, av_q_len_station6, av_q_len_station7, av_q_len_station8, av_q_len_station9, av_q_len_station10];

    % save('average_queue_length_trace.mat','trace_1');

    save(['average_queue_length_trace_' num2str(i) '.mat'],'average_queue_length_trace')
    
end 

