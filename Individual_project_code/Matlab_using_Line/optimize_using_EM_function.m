clear all

var_map_generated = Inf;
S = 2;
N = 3;

MAP_random = map_rand(N+1);
D1 = MAP_random{2};

while true
    
    [exp_max, mean_map_real, var_map_real, mean_map_generated, var_map_generated] = EM_algorithm_function(S,N,D1);
    
    if var_map_generated ~= Inf && abs(var_map_real - var_map_generated) <= 0.3 && abs(mean_map_real - mean_map_generated) <= 0.3
        break 
    end   
end 


generated_MAP = exp_max
real_mean = mean_map_real
generated_mean = mean_map_generated
real_var = var_map_real
generated_var = var_map_generated

