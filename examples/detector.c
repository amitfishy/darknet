#include "darknet.h"

void train_detector(char *cfg_file_path, char *annotation_folder_path, char *output_folder_path, char *train_sets_path, int max_batches, char *weights_file_path, char *output_model_filename, int *gpus, int ngpus, int clear)
{
    //list *options = read_data_cfg(datacfg);
    char *output_directory = output_folder_path;

    srand(time(0));
    //printf("%s\n", cfg_file_path);
    //printf("%s\n", output_model_filename);

    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfg_file_path, weights_file_path, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1];
    net->max_batches = get_current_batch(net) + max_batches;

    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_sets_path);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);
    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.annotation_folder = annotation_folder_path;
    //args.type = INSTANCE_DATA;
    args.threads = 64;

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net->max_batches){
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
//Uncomment this block to store snapshots
/*        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[1024];
            sprintf(buff, "%s/%s.backup", output_directory, output_model_filename);
            save_weights(net, buff);
        }
        if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[1024];
            sprintf(buff, "%s/%s_%d.weights", output_directory, output_model_filename, i);
            save_weights(net, buff);
        }*/
        free_data(train);
    }
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[1024];
    sprintf(buff, "%s/%s", output_directory, output_model_filename);
    save_weights(net, buff);
}

void print_detector_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2. + 1;
        float xmax = boxes[i].x + boxes[i].w/2. + 1;
        float ymin = boxes[i].y - boxes[i].h/2. + 1;
        float ymax = boxes[i].y + boxes[i].h/2. + 1;

        /*
        //1-base indices - for PASCAL VOC evaluation
        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;*/
        
        //0-base indices - for KITTI evaluation
        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w-1) xmax = w-1;
        if (ymax > h-1) ymax = h-1;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector(char *cfg_file_path, char *class_names_file_path, char *output_results_folder_path, char *test_sets_path, char *weights_file_path)
{
    int j;    
    char **names = get_labels(class_names_file_path);
    char *mapf = 0;
    int *map = 0;
    if (mapf) map = read_map(mapf);
    
    network *net = load_network(cfg_file_path, weights_file_path, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(test_sets_path);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024], output_file_name[512];
    FILE **fps = 0;

    strcpy(output_file_name, "comp4_det_");
    if (strcmp(basecfg(test_sets_path), "val")==0)
        strcat(output_file_name, "val_");
    else if(strcmp(basecfg(test_sets_path), "test")==0)
        strcat(output_file_name, "test_");

    fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j)
    {
        snprintf(buff, 1024, "%s/%s%s.txt", output_results_folder_path, output_file_name, names[j]);
        fps[j] = fopen(buff, "w");
    }

    box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
    float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(classes+1, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .01;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "darknet - Detected in Image: %d/%d\n", i, m);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            get_region_boxes(l, w, h, net->w, net->h, thresh, probs, boxes, 0, 0, map, .5, 0);
            if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, classes, nms);
            print_detector_detections(fps, id, boxes, probs, l.w*l.h*l.n, classes, w, h);

            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }

    fprintf(stderr, "darknet - Total Detection Time: %fs\n", what_time_is_it_now() - start);
}

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.3;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];

        box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes + 1, sizeof(float *));
        float **masks = 0;
        if (l.coords > 4){
            masks = calloc(l.w*l.h*l.n, sizeof(float*));
            for(j = 0; j < l.w*l.h*l.n; ++j) masks[j] = calloc(l.coords-4, sizeof(float *));
        }

        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        get_region_boxes(l, im.w, im.h, net->w, net->h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
        if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        //else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, masks, names, alphabet, l.classes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            cvNamedWindow("predictions", CV_WINDOW_NORMAL); 
            if(fullscreen){
                cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            }
            show_image(im, "predictions");
            cvWaitKey(0);
            cvDestroyAllWindows();
#endif
        }

        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
        if (filename) break;
    }
}

void run_detector(int argc, char **argv)
{
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");

    //General Parameters for training and testing images (and some for running over video as well)
    //Should contain valid file path
    char *cfg_file_path = find_char_arg(argc, argv, "-net_config_file", 0);
    //Class Names to be classified
    char *class_names_file_path = find_char_arg(argc, argv, "-class_names_file", 0);
    //Modified Annotations folder
    char *annotation_folder_path = find_char_arg(argc, argv, "-annotation_folder", 0);
    folder_proper_format(annotation_folder_path);
    //Trained models stored here
    char *output_folder_path = find_char_arg(argc, argv, "-output_model_folder", 0);
    folder_proper_format(output_folder_path);
    //Test results stored 
    char *output_results_folder_path = find_char_arg(argc, argv, "-output_results_folder", 0);
    folder_proper_format(output_results_folder_path);
    //Initial Weight Model
    char *weights_file_path = find_char_arg(argc, argv, "-weights_file", 0);
    //1-train is on. 0-train is off. Default-0.
    int train = find_int_arg(argc, argv, "-train", 0);
    char *train_sets_path = find_char_arg(argc, argv, "-train_sets_file", 0);
    //max_batches is the same as num_iterations.
    int max_batches = find_int_arg(argc, argv, "-max_batches", 0);
    //Trained Model is output with output_model_prefix + number of iterations (max batches)
    char *output_model_filename = find_char_arg(argc, argv, "-output_model_filename", 0);
    //1-test is on. 0-test is off. Default-0. test only produces result file. Evaluation is done with python script (same as faster-rcnn).
    int test = find_int_arg(argc, argv, "-test", 0);
    char *test_sets_path = find_char_arg(argc, argv, "-test_sets_file", 0);

    //General Parameters    
    printf("Network Config File: `%s`\n", cfg_file_path);
    printf("Class Names File: `%s`\n", class_names_file_path);
    printf("Annotation Folder (temp generated): `%s`\n", annotation_folder_path);
    printf("Output Folder (store models): `%s`\n", output_folder_path);
    printf("Output Results Folder: `%s`\n", output_results_folder_path);
    printf("Weights File: `%s`\n", weights_file_path);
    printf("Train Sets (temp generated): `%s`\n", train_sets_path);
    printf("Trained Model: `%s`\n", output_model_filename);
    printf("Test Sets (temp generated): `%s`\n", test_sets_path);

    if (train==1)
    {   
        train_detector(cfg_file_path, annotation_folder_path, output_folder_path, train_sets_path, max_batches, weights_file_path, output_model_filename, gpus, ngpus, clear);
    }
    if (test==1)
    {
        validate_detector(cfg_file_path, class_names_file_path, output_results_folder_path, test_sets_path, weights_file_path);
    }

}
