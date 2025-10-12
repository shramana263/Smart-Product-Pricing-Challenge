
# Training Template (to be executed with GPU)

def train_distilbert_model(train_df, target_col, output_dir, config):
    '''Train DistilBERT for regression on transformed target'''
    
    # 1. Prepare data
    tokenizer = DistilBertTokenizer.from_pretrained(config['model_name'])
    
    def tokenize_function(examples):
        return tokenizer(
            examples['catalog_content'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=config['max_length']
        )
    
    # 2. Create dataset
    from datasets import Dataset
    dataset = Dataset.from_pandas(train_df[['catalog_content', target_col]])
    dataset = dataset.map(tokenize_function, batched=True)
    
    # 3. Define model
    from transformers import DistilBertForSequenceClassification
    model = DistilBertForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=1  # Regression
    )
    
    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['epochs'],
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=True,  # Mixed precision
        dataloader_num_workers=4,
    )
    
    # 5. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics_regression
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    
    return trainer

# Execute training for each transformation
for target_name in ['target_log', 'target_sqrt', 'target_boxcox']:
    print(f'\nTraining model for {target_name}...')
    output_dir = CONFIG['output_dir'] / target_name
    
    # Train on 4 folds, validate on 1
    for fold in range(CONFIG['n_folds']):
        train_df = train[train['fold'] != fold]
        val_df = train[train['fold'] == fold]
        
        trainer = train_distilbert_model(
            train_df, 
            target_name, 
            output_dir / f'fold_{fold}',
            CONFIG
        )
        
        # Validate
        predictions = trainer.predict(val_df)
        
        # Inverse transform
        if target_name == 'target_log':
            y_pred = np.expm1(predictions)
        elif target_name == 'target_sqrt':
            y_pred = predictions ** 2
        else:  # boxcox
            y_pred = pt.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # Calculate SMAPE
        fold_smape = smape(val_df['price'].values, y_pred)
        print(f'  Fold {fold} SMAPE: {fold_smape:.2f}%')
