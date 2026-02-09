import tensorflow as tf

tfrecord_path = "/media/raid/workspace/tengbo/any4lerobot/data/rlds_data/InterceptMedium-v1/mikasa_dataset/1.0.0/mikasa_dataset-train.tfrecord-00000-of-00001"

dataset = tf.data.TFRecordDataset(tfrecord_path)

for i, record in enumerate(dataset.take(2)):
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    
    print(f"\n=== Example {i} ===")
    features = list(example.features.feature.keys())
    print(f"Total features: {len(features)}")
    print(f"First 15 features: {features[:15]}")
    
    # Check observation and action structure
    for key in ['observation/state', 'action', 'observation/image', 'reward', 'is_terminal', 'episode_id', 'timestamp', 'step_metadata']:
        if key in features:
            f = example.features.feature[key]
            if f.HasField('float_list'):
                vals = f.float_list.value
                print(f"  {key}: float_list len={len(vals)}, first={vals[:3] if len(vals)<=10 else vals[:3]}...")
            elif f.HasField('int64_list'):
                vals = f.int64_list.value
                print(f"  {key}: int64_list len={len(vals)}, values={vals}")
            elif f.HasField('bytes_list'):
                vals = f.bytes_list.value
                print(f"  {key}: bytes_list len={len(vals)}")
