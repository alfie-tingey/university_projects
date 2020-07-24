function [exp_max, mean_map_real, var_map_real, mean_map_generated, var_map_generated] = EM_algorithm_function(S,N,D1)
    
    model = Network('MIP');
    
    %% Block 1: nodes
    delay = Delay(model,'WorkingState');
    queue = Queue(model, 'RepairQueue', SchedStrategy.FCFS);
    queue.setNumberOfServers(S);
    %% Block 2: classes
    cclass = ClosedClass(model, 'Machines', N, delay);
    delay.setService(cclass, Exp(0.5));
    queue.setService(cclass, Exp(4.0));

    %% Block 3: topology
    model.link(Network.serialRouting(delay,queue));

    % Additional stuff for Giuliano

    % model.jsimgView
    % model = Network('M/G/1');
    solver = SolverCTMC(model);
    ctmcAvgTable = solver.getAvgTable;
    StateSpace = solver.getStateSpace();
    InfGen = full(solver.getGenerator());
    Q = InfGen;

    %% Set matrices D1 and D0 from Q which we will use for creating an MAP that we will aim to generate using EM-algorithm
    % D1= [0,0,0,1.000;  0.5000,0,0,0; 0,1.0,0,0; 0,0,1.5,0];
    D0 = Q - D1;
    % D0=Q-D1;
    MAP={D0,D1};
    plot(map_acf(MAP,1:100));
    map_mean(MAP);
    map_var(MAP);
    % map_sample(MAP,1e3);
    S = map_sample(MAP,1e3);

    %% Work out pi and tau for the original ctmc
    pi = map_prob(MAP);
    tau = map_pie(MAP);
    pi_dash = (pi * D1)./(pi * D1 * ones(N+1,1));

    %% START OF EM ALGORITHM STUFF ETC %%


    %% Likelihood functions and EM algorithm attempts using the sample S as the trace. Randomely choose P0 and P1

    % Define the trace as a sample from the MAP defined in previous section
    trace = S;

    % Find the length of the trace
    size_trace = size(trace);
    length_trace = size_trace(1);

    % define the alpha constant as the minimum inverted value of the trace

    % alpha_start = min((trace).^(-1))
    alpha_start = 1.9*max(abs(diag(D0)));

    % Define the matrices P0 and P1 randomly such that P0+P1 is stochastic. Use
    % function from 'Roger Stafford (2020). Random Vectors with Fixed Sum'.

    % P0 = [0.2,0.1,0.2,0.1;0.1,0.1,0.2,0.1;0.2,0.1,0.1,0.1;0.1,0.2,0.2,0.1];
    % P1 = [0.1,0.1,0.1,0.1;0.1,0.2,0.1,0.1;0.1,0.1,0.2,0.1;0.1,0.1,0.1,0.1];

    P0 = randfixedsum(N+1,N+1,0.5,0.1,1)';
    P1 = randfixedsum(N+1,N+1,0.5,0.1,1)';
    
    % define a variable to let us see what the random P matrices are
    P_reveal_0 = P0;
    P_reveal_1 = P1;

    % Set P0old and P1old to something arbitrary such that we change them later
    % during the EM-algorithm

    P0old = 4*ones(N+1,N+1);
    P1old = 4*ones(N+1,N+1);

    % Start while loop which evaluates the conditions on P0 and P1

    %% Start the EM-Algorithm
    
    iter = 0;
    
    lastwarn('')


    while true

        % Set a_0 to be pi_dash as defined in the algorithm.
        
        iter = iter + 1

        D1_dash = P1*alpha_start;
        I = eye(N+1,N+1);
        D0_dash = (P0-I)*alpha_start;

        MAP_dash = {D0_dash,D1_dash};
        pi = map_prob(MAP_dash);
        pi_dash = (pi * D1_dash)./(pi * D1_dash * ones(N+1,1));

        a_old = pi_dash;
        % set b_m as just e^T
        b_m = ones(N+1,1);

        % Put the likelihood vectors in double format and normalize them
        a_old = normalize(double(a_old), 'norm');
        b_m = normalize(double(b_m), 'norm');

        list_a = zeros(length_trace,N+1);
        list_b = zeros(N+1,length_trace);

        list_a(1,:) = a_old;
        list_b(:,length_trace) = b_m;

        %% Find likelihood vectors 

        % Create a for loop to find all of the subsequent 'a' likelihood vectors.
        for i = 2:length_trace
            % pre-define a_new as 0 which we input into the sum.
            D0_ti = expm(D0_dash.*trace(i));
            % Find a_new from equation (1) of Buchholz paper.
            a_new = a_old*D0_ti*P1;
            % Normalize the likelihood vector a_new
            a_new = normalize(double(a_new), 'norm');
            % create a 'list' to store all of the likelihood vectors for a. 
            list_a(i,:) = a_new;
            % Set a_old as a_new. 
            a_old = a_new;
        end

        % Now use a for loop to find the 'b' vectors.
        for i = 1:length_trace-1
            D0_ti = expm(D0_dash.*trace(i));
            % Find all of the b_m values using equation (2) in Buccholz.
            b_new = D0_ti*P1*b_m;
            b_new = normalize(double(b_new), 'norm');
            list_b(:,length_trace - i) = b_new;
            b_m = b_new;
        end

        %% Find the forward and backward likelihoods v and w

        % In this section of the algorithm we want to find the forward and
        % backward likelihoods, v and w respectively, as we need them to find
        % the matrices X0 and X1.

        % Set the values for truncation. Can change these... if l_i < 3 then
        % we get some NAN values in the matrices P0 and P1 which ruins the
        % algorithm.

        l_i = 4;
        r_i = 100;


        % Pre-define the size of vs and ws using the length of the trace. 
        list_vs = zeros(length_trace, N+1,r_i);
        list_ws = zeros(N+1,length_trace,l_i);

        % Start a nested for loop to work out forward likelihoods:
        for i = 1:length_trace
            for l = 1:r_i
                % Use list_a in multiples of 4, as that is how I have defined
                % the list to be. Each 4 values corresponds to a likelihood.
                % Work out forward likelihoods from equation in Buchholz. 
                vs = list_a(i,:)*(P0)^(l);
                list_vs(i,:,l) = vs;
            end 
        end

        % Do a similar nested loop for the backward likelihoods. Again, I
        % should put these two in the same loop. 

        for i = 1:length_trace-1
            for l = 1:l_i
                ws = (P0)^l*P1*list_b(:,i+1);
                list_ws(:,i,l) = ws;
            end 
        end

        %% Compute the right betas 

        % We need to work out the beta values to put them into the algorithm to
        % find matrices X0 and X1. We are trying to compute equations (4) and
        % (5) from Buchholz paper. 

        % Set the truncation points (change these when we hear back from Giuliano)

        % Formula for the betas: beta(k,alpha*t) =
        % exp(-alpha*t)(alpha*t)*k/factorial(k).

        % Pre-define size of beta matrix.

        beta = zeros(length_trace, r_i-l_i);

        for i = 1:length_trace
            for j = 1:r_i
                % Find beta. Truncation points need to be altered after hearing
                % back from Giuliano.
                beta(i,j) = exp(-alpha_start*trace(i))*(alpha_start*trace(i))^(j)/factorial(j);
            end
        end    

        %% Compute X0

        % Here is our computation of equation (4) in Buchholz paper. 

        % Pre-define X0 size. We make it a 3 dimensional matrix, as each 2
        % dimensional matrix corresponds to an X0^(i).

        X0 = zeros(N+1,N+1,length_trace);

        % Create a nested loop for the computation. The size of our P matrices
        % are 4x4, but in the next tuning of this algorithm we can set the size
        % of the loop to correspond to the size of the matrices. 

        for x = 1:N+1
            for y = 1:N+1
                for i = 1:length_trace
                    % pre-define a list of each matrix, as eventually we will
                    % need to sum them.
                    beta_sum = sum(beta(i,l_i:r_i-1));
                    likelihood_value = 0;
                    for l = 1:l_i-1
                        vs_value = list_vs(i,:,l);
                        if i < length_trace
                            ws_value = list_ws(:,i+1,l_i-l);
                        else
                            ws_value = list_ws(:,i,l_i-l);
                        end
                        likelihood_value = likelihood_value + vs_value(x)*P0(x,y)*ws_value(y);
                    %likelihood_sum = sum(
                    end
                    X0(x,y,i) = beta_sum*likelihood_value;
                end
            end
        end

        %% Compute X1

        % Similar explanation as when computing X0.

        X1 = zeros(N+1,N+1,length_trace);

        for x = 1:N+1
            for y = 1:N+1
                for i = 1:length_trace-1
                    % pre-define a list of each matrix, as eventually we will
                    % need to sum them.
                    likelihood_value = 0;
                    for l = l_i:r_i-1
                        vs_value = list_vs(i,:,l);
                        list_b_vector = list_b(:,i+1);
                        likelihood_value = likelihood_value + beta(i,l)*vs_value(x)*P1(x,y)*list_b_vector(y);
                    %likelihood_sum = sum(
                    end
                    X1(x,y,i) = likelihood_value;
                end
            end
        end

        %% Compute Y0 and Y1


        % Compute and normalize Y matrices using equations from Buchholz.

        Y0 = zeros(N+1,N+1);
        Y1 = zeros(N+1,N+1);

        for i = 1:length_trace
            Y0 = Y0 + X0(:,:,i);
            Y1 = Y1 + X1(:,:,i);
        end 

        % Find X0s

        Y_normal_1 = zeros(N+1,N+1);
        Y_normal_0 = zeros(N+1,N+1);

        for x = 1:N+1
            for y = 1:N+1
                if abs(sum(Y0(x,:))+sum(Y1(x,:))) < 0.01
                    Y_normal_0(x,y) = Y0(x,y);
                    Y_normal_1(x,y) = Y1(x,y);
                else
                    Y_normal_0(x,y) = Y0(x,y)/(sum(Y0(x,:))+sum(Y1(x,:)));
                    Y_normal_1(x,y) = Y1(x,y)/(sum(Y0(x,:))+sum(Y1(x,:)));
                end 
            end
        end

        P0old = P0;
        P1old = P1;

        P0 = Y_normal_0;
        P1 = Y_normal_1;
        
        [warnmsg, msgid] = lastwarn;
        if strcmp(msgid,'MATLAB:singularMatrix')
            disp('Warning has occurred')
            break
        end

        % Calculate the 'max difference' so we know when to stop the
        % algorithm.

        if all(max(abs(P0 - P0old),abs(P1 - P1old)) < 0.005)
            break 
        end 

    end 


    %% Final stuff to create MAP from expected maximum algorithm

    D0_generated = alpha_start*(P0 - diag(P0*ones(N+1,1)+P1*ones(N+1,1)));
    D1_generated = alpha_start*P1;

    MAP_algorithm = {D0_generated,D1_generated};
    MAP_algorithm = map_normalize(MAP_algorithm);
    
    exp_max = MAP_algorithm;

    %% Have a look at the metrics of the MAPs to see performance
    
    if strcmp(msgid,'MATLAB:singularMatrix')
        var_map_generated = 0;
        mean_map_generated = 0;
    else
        var_map_generated = map_var(MAP_algorithm);
        mean_map_generated = map_mean(MAP_algorithm);
    end

    mean_map_real = map_mean(MAP);
%     mean_map_generated = map_mean(MAP_algorithm);
    
    var_map_real = map_var(MAP);
%     var_map_generated = map_var(MAP_algorithm);

end

