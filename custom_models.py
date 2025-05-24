import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from custom_utils import initialize_crf_transitions_bio, get_rank_safe

class CustomCRF(nn.Module):
    """
    Оптимизированная реализация CRF (Conditional Random Field) для NER-задач.
    """
    def __init__(self, num_tags, batch_first=True):
        super(CustomCRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def _get_transitions_info(self):
        """
        Получение информации о переходах для отладки или логирования
        """
        if not hasattr(self, 'transitions'):
            return {}
            
        transitions_stats = {
            "transitions_mean": self.transitions.mean().item(),
            "transitions_std": self.transitions.std().item(),
            "transitions_min": self.transitions.min().item(),
            "transitions_max": self.transitions.max().item(),
        }
        if hasattr(self, 'start_transitions'):
            transitions_stats.update({
                "start_transitions_mean": self.start_transitions.mean().item(),
                "start_transitions_std": self.start_transitions.std().item(),
            })
        if hasattr(self, 'end_transitions'):
            transitions_stats.update({
                "end_transitions_mean": self.end_transitions.mean().item(),
                "end_transitions_std": self.end_transitions.std().item(),
            })
        return transitions_stats
        
    def copy_transitions_from(self, source_crf):
        """
        Копирует матрицы переходов из другого CRF слоя
        """
        if hasattr(source_crf, 'transitions') and hasattr(self, 'transitions'):
            if source_crf.transitions.shape == self.transitions.shape:
                self.transitions.data.copy_(source_crf.transitions.data)
                print(f"Матрица transitions скопирована. Форма: {self.transitions.shape}")
            else:
                print(f"Невозможно скопировать transitions: разные размеры ({source_crf.transitions.shape} vs {self.transitions.shape})")
                
        if hasattr(source_crf, 'start_transitions') and hasattr(self, 'start_transitions'):
            if source_crf.start_transitions.shape == self.start_transitions.shape:
                self.start_transitions.data.copy_(source_crf.start_transitions.data)
                print(f"Матрица start_transitions скопирована. Форма: {self.start_transitions.shape}")
            else:
                print(f"Невозможно скопировать start_transitions: разные размеры")
                
        if hasattr(source_crf, 'end_transitions') and hasattr(self, 'end_transitions'):
            if source_crf.end_transitions.shape == self.end_transitions.shape:
                self.end_transitions.data.copy_(source_crf.end_transitions.data)
                print(f"Матрица end_transitions скопирована. Форма: {self.end_transitions.shape}")
            else:
                print(f"Невозможно скопировать end_transitions: разные размеры")

    def _forward_alg(self, feats, mask=None):
        if not self.batch_first:
            feats = feats.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        batch_size, seq_length, num_tags = feats.size()
        
        log_alpha = torch.full((batch_size, num_tags), -10000.0, device=feats.device) 
        log_alpha[:, self.num_tags -2] = 0.
                                          
        current_log_alpha = self.start_transitions + feats[:, 0]

        for i in range(1, seq_length):
            alpha_t_minus_1 = current_log_alpha.unsqueeze(2)
            transitions_t = self.transitions.unsqueeze(0).expand(batch_size, -1, -1)
            emissions_t = feats[:, i].unsqueeze(1)
            scores = alpha_t_minus_1 + transitions_t + emissions_t
            new_log_alpha = torch.logsumexp(scores, dim=1)
            
            if mask is not None:
                mask_i = mask[:, i].unsqueeze(-1).bool()
                current_log_alpha = torch.where(mask_i, new_log_alpha, current_log_alpha)
            else:
                current_log_alpha = new_log_alpha

        terminal_vars = current_log_alpha + self.end_transitions.unsqueeze(0)
        log_partition = torch.logsumexp(terminal_vars, dim=-1)
        return log_partition

    def _score_sentence(self, feats, tags, mask=None):
        if not self.batch_first:
            feats = feats.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)

        batch_size, seq_length, _ = feats.size()
        score = torch.zeros(batch_size, device=feats.device)

        first_tags = tags[:, 0]
        score += self.start_transitions.gather(0, first_tags)
        score += feats[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze(1)

        for i in range(1, seq_length):
            from_tags = tags[:, i - 1]
            to_tags = tags[:, i]
            
            transition_scores = self.transitions[to_tags, from_tags]
            emission_scores = feats[:, i].gather(1, to_tags.unsqueeze(1)).squeeze(1)
            
            if mask is not None:
                mask_i = mask[:, i].bool()
                score += (transition_scores + emission_scores) * mask_i
            else:
                score += transition_scores + emission_scores

        if mask is not None:
            seq_ends = mask.sum(dim=1).long() - 1
            last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        else:
            last_tags = tags[:, -1]
        
        score += self.end_transitions.gather(0, last_tags)
        return score

    def forward(self, emissions, tags, mask=None, reduction='mean'):
        if mask is None and emissions is not None:
            mask = torch.ones_like(tags, dtype=torch.bool, device=emissions.device)
        elif mask is None:
            raise ValueError("Mask and emissions cannot both be None in CRF forward")
        
        if mask.size(1) > 0:
            mask_clone = mask.clone()
            mask_clone[:, 0] = True
        else:
            mask_clone = mask

        gold_score = self._score_sentence(emissions, tags, mask_clone)
        log_partition = self._forward_alg(emissions, mask_clone)
        loss = log_partition - gold_score

        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            if loss.numel() == 0:
                return torch.tensor(0.0, device=loss.device, requires_grad=True)
            return loss.mean()
        elif reduction == 'token_mean':
            num_tokens = mask_clone.sum()
            if num_tokens == 0:
                 return torch.tensor(0.0, device=loss.device, requires_grad=True)
            return loss.sum() / num_tokens
        else:
            raise ValueError(f"Unknown reduction type: {reduction}")

    def decode(self, emissions, mask=None):
        if not self.batch_first:
            emissions = emissions.transpose(0,1)
            if mask is not None:
                mask = mask.transpose(0,1)
        
        rank_safe = get_rank_safe()

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        
        if mask.dtype != torch.bool:
            mask = mask.bool()

        if mask.size(1) > 0:
            mask_clone = mask.clone()
            mask_clone[:, 0] = True
        else:
            mask_clone = mask
        
        batch_size, seq_length, num_tags = emissions.size()

        if seq_length == 0:
            return [[] for _ in range(batch_size)]

        current_scores = self.start_transitions + emissions[:, 0] 
        history = []

        for i in range(1, seq_length):
            prev_scores_expanded = current_scores.unsqueeze(2)
            transitions_expanded = self.transitions.unsqueeze(0)
            possible_scores = prev_scores_expanded + transitions_expanded
            best_prev_scores, best_prev_tags = torch.max(possible_scores, dim=1)
            current_emissions = emissions[:, i]
            new_scores = best_prev_scores + current_emissions
            history.append(best_prev_tags)
            
            mask_i = mask_clone[:, i].unsqueeze(-1).bool() 
            current_scores = torch.where(mask_i, new_scores, current_scores)

        current_scores += self.end_transitions.unsqueeze(0) 
        best_overall_scores, last_best_tags = torch.max(current_scores, dim=1) 
        
        best_paths = []
        for b in range(batch_size):
            actual_seq_len = mask_clone[b].sum().item()

            if actual_seq_len == 0: 
                best_paths.append([])
                continue

            current_best_tag_val = last_best_tags[b].item()
            path = [current_best_tag_val]
            
            for hist_idx in range(actual_seq_len - 2, -1, -1):
                hist_entry = history[hist_idx]
                current_best_tag_val = hist_entry[b, current_best_tag_val].item()
                path.append(current_best_tag_val)
            path.reverse()
            best_paths.append(path)
        
        return best_paths


class SpanBertForTokenClassificationCRF(nn.Module):
    def __init__(self, base_model_path, num_labels, config, dropout_rate=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.config = config
        
        # Обновляем конфигурацию для совместимости, если атрибуты еще не заданы
        if not hasattr(self.config, "architectures") or not self.config.architectures:
            self.config.architectures = ["SpanBertForTokenClassificationCRF"]
        elif "SpanBertForTokenClassificationCRF" not in self.config.architectures:
            self.config.architectures = ["SpanBertForTokenClassificationCRF"]
            
        if not hasattr(self.config, "model_type"):
            self.config.model_type = "bert-crf"
        if not hasattr(self.config, "has_crf"):
            self.config.has_crf = True
        if not hasattr(self.config, "crf_num_tags"):
            self.config.crf_num_tags = num_labels
        if not hasattr(self.config, "dropout_rate"):
            self.config.dropout_rate = dropout_rate

        try:
            # Сначала пытаемся загрузить с safetensors
            self.bert = AutoModel.from_pretrained(base_model_path, config=config, use_safetensors=True)
        except Exception as e:
            print(f"Ошибка при загрузке модели с safetensors: {e}")
            try:
                # Пробуем загрузить как TF модель
                self.bert = AutoModel.from_pretrained(base_model_path, config=config, from_tf=True)
                print(f"Модель успешно загружена с from_tf=True")
            except Exception as e2:
                print(f"Ошибка при загрузке модели с from_tf=True: {e2}")
                # В крайнем случае пробуем другие варианты
                try:
                    # Создаем модель с нуля на основе конфигурации
                    self.bert = AutoModel.from_config(config)
                    print(f"Модель создана с нуля на основе конфигурации")
                except Exception as e3:
                    print(f"Критическая ошибка при инициализации модели: {e3}")
                    raise RuntimeError("Не удалось инициализировать базовую модель")
                
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CustomCRF(num_labels, batch_first=True)

        self.id2label = config.id2label if hasattr(config, "id2label") else {i: f"LABEL_{i}" for i in range(num_labels)}
        self.label2id = config.label2id if hasattr(config, "label2id") else {f"LABEL_{i}": i for i in range(num_labels)}
        
        initialize_crf_transitions_bio(self.crf, self.id2label, self.label2id, is_custom_crf=True)

    @classmethod
    def from_pretrained(cls, model_path, *args, **kwargs):
        """
        Загрузка модели из сохраненной директории
        """
        import os
        
        config = kwargs.get('config', None)
        if config is None:
            from transformers import AutoConfig
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                config = AutoConfig.from_pretrained(model_path)
                kwargs['config'] = config
            else:
                raise ValueError(f"Не найден config.json в {model_path}")
        
        # Получаем параметры из конфигурации или используем значения по умолчанию
        dropout_rate = getattr(config, 'dropout_rate', 0.1)
        num_labels = config.num_labels
        
        # Создаем экземпляр модели
        model = cls(
            base_model_path=model_path, 
            num_labels=num_labels, 
            config=config,
            dropout_rate=dropout_rate,
        )
        
        # Загружаем состояние модели
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"Веса модели загружены из {weights_path}")
        else:
            print(f"ВНИМАНИЕ: Не найден файл весов pytorch_model.bin в {model_path}")
        
        return model
        
    def save_pretrained(self, output_dir):
        """
        Сохранение модели в формате huggingface
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем конфигурацию
        self.config.save_pretrained(output_dir)
        
        # Сохраняем веса модели
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Сохраняем словарь меток
        import json
        labels_path = os.path.join(output_dir, "label_map.json")
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump({
                "id2label": self.id2label,
                "label2id": self.label2id
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Модель успешно сохранена в {output_dir}")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_hidden_states=False,
        class_weights=None,
        **kwargs,
    ):
        # if hasattr(self.bert, "gradient_checkpointing_disable") and self.bert.training:
        #     self.bert.gradient_checkpointing_disable()

        bert_kwargs = {}
        if token_type_ids is not None: bert_kwargs['token_type_ids'] = token_type_ids

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **bert_kwargs,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            crf_mask = (labels != -100).bool()
            
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = self.label2id.get('O', 0)

            try:
                loss = self.crf(logits, crf_labels, mask=crf_mask, reduction="mean")
            except ValueError as e:
                if get_rank_safe() <= 0:
                    print(f"Ошибка CRF (SpanBertForTokenClassificationCRF): {e}")
                    print(f"Размерность logits: {logits.shape}, crf_labels: {crf_labels.shape}, crf_mask: {crf_mask.shape}")
                    print(f"Min/Max в crf_labels: {crf_labels.min()}, {crf_labels.max()}")
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not output_hidden_states:
            return {"loss": loss, "logits": logits}
        else:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": outputs.hidden_states,
            }

    def decode(self, logits, mask):
        """Декодирование последовательности с использованием CRF"""
        return self.crf.decode(logits, mask=mask.bool() if mask is not None else None)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enables gradient checkpointing for the base BERT model."""
        if hasattr(self.bert, "gradient_checkpointing_enable"):
            self.bert.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self):
        """Disables gradient checkpointing for the base BERT model."""
        if hasattr(self.bert, "gradient_checkpointing_disable"):
            self.bert.gradient_checkpointing_disable()


class SpanBertLSTMForTokenClassification(nn.Module):
    def __init__(
        self,
        base_model_path,
        num_labels,
        config,
        lstm_hidden_size=768,
        dropout_rate=0.1,
        bidirectional=True,
        attention_dim=256,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.config = config
        
        # Сохраняем и обновляем параметры LSTM в конфигурации, если они еще не заданы
        if not hasattr(self.config, "lstm_hidden_size"):
            self.config.lstm_hidden_size = lstm_hidden_size
        if not hasattr(self.config, "lstm_bidirectional"):
            self.config.lstm_bidirectional = bidirectional
        if not hasattr(self.config, "lstm_num_layers"):
            self.config.lstm_num_layers = 2
        # Используем переданный dropout_rate для кастомных слоев и сохраняем его
        self.config.custom_model_dropout_rate = dropout_rate
        if not hasattr(self.config, "lstm_dropout"):
            self.config.lstm_dropout = 0.3 if self.config.custom_model_dropout_rate > 0.1 else 0
        if not hasattr(self.config, "attention_dim"):
            self.config.attention_dim = attention_dim
        if not hasattr(self.config, "crf_enabled"):
            self.config.crf_enabled = True
        
        # Убедимся, что architectures и model_type заданы правильно
        self.config.architectures = ["SpanBertLSTMForTokenClassification"]
        self.config.model_type = "bert-lstm-crf"

        try:
            # Сначала пытаемся загрузить с safetensors
            self.bert = AutoModel.from_pretrained(base_model_path, config=config, use_safetensors=True)
        except Exception as e:
            print(f"Ошибка при загрузке модели с safetensors: {e}")
            try:
                # Пробуем загрузить как TF модель
                self.bert = AutoModel.from_pretrained(base_model_path, config=config, from_tf=True)
                print(f"Модель успешно загружена с from_tf=True")
            except Exception as e2:
                print(f"Ошибка при загрузке модели с from_tf=True: {e2}")
                # В крайнем случае пробуем другие варианты
                try:
                    # Создаем модель с нуля на основе конфигурации
                    self.bert = AutoModel.from_config(config)
                    print(f"Модель создана с нуля на основе конфигурации")
                except Exception as e3:
                    print(f"Критическая ошибка при инициализации модели: {e3}")
                    raise RuntimeError("Не удалось инициализировать базовую модель")
        
        # Используем значения из объекта config, если они там уже есть
        lstm_hidden_size = getattr(self.config, "lstm_hidden_size", lstm_hidden_size)
        lstm_num_layers = getattr(self.config, "lstm_num_layers", 2)
        lstm_dropout = getattr(self.config, "lstm_dropout", 0.3 if dropout_rate > 0.1 else 0)
        bidirectional = getattr(self.config, "lstm_bidirectional", bidirectional)
        
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers, 
            batch_first=True,
            dropout=lstm_dropout, 
            bidirectional=bidirectional,
        )

        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size

        self.attention_layer = nn.Sequential(
            nn.Linear(lstm_output_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

        self.transform_layer = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.LayerNorm(lstm_output_size),
            nn.GELU(),
            nn.Dropout(self.config.custom_model_dropout_rate),
        )

        self.dropout = nn.Dropout(self.config.custom_model_dropout_rate)
        self.classifier = nn.Linear(lstm_output_size, num_labels)

        self.crf = CustomCRF(num_labels, batch_first=True) 
        self.id2label = config.id2label if hasattr(config, "id2label") else {i: f"LABEL_{i}" for i in range(num_labels)}
        self.label2id = config.label2id if hasattr(config, "label2id") else {f"LABEL_{i}": i for i in range(num_labels)}
        initialize_crf_transitions_bio(self.crf, self.id2label, self.label2id, is_custom_crf=True)
        
    @classmethod
    def from_pretrained(cls, model_path, *args, **kwargs):
        """
        Загрузка модели из сохраненной директории
        """
        import os
        
        config = kwargs.get('config', None)
        if config is None:
            from transformers import AutoConfig
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                config = AutoConfig.from_pretrained(model_path)
                kwargs['config'] = config
            else:
                raise ValueError(f"Не найден config.json в {model_path}")
        
        # Получаем параметры LSTM из конфигурации или используем значения по умолчанию
        lstm_hidden_size = getattr(config, 'lstm_hidden_size', 768)
        bidirectional = getattr(config, 'lstm_bidirectional', True)
        attention_dim = getattr(config, 'attention_dim', 256)
        # Используем сохраненный custom_model_dropout_rate, если он есть
        # Иначе, используем hidden_dropout_prob из базовой модели, или 0.1 как крайний случай
        dropout_rate_to_use = getattr(config, 'custom_model_dropout_rate', 
                                    getattr(config, 'hidden_dropout_prob', 0.1))
        num_labels = config.num_labels
        
        # Создаем экземпляр модели
        model = cls(
            base_model_path=model_path, 
            num_labels=num_labels, 
            config=config,
            lstm_hidden_size=lstm_hidden_size,
            bidirectional=bidirectional,
            attention_dim=attention_dim,
            dropout_rate=dropout_rate_to_use,
        )
        
        # Загружаем состояние модели
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"Веса модели загружены из {weights_path}")
        else:
            print(f"ВНИМАНИЕ: Не найден файл весов pytorch_model.bin в {model_path}")
        
        return model
        
    def save_pretrained(self, output_dir):
        """
        Сохранение модели в формате huggingface
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем конфигурацию
        self.config.save_pretrained(output_dir)
        
        # Сохраняем веса модели
        model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Сохраняем словарь меток
        import json
        labels_path = os.path.join(output_dir, "label_map.json")
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump({
                "id2label": self.id2label,
                "label2id": self.label2id
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Модель успешно сохранена в {output_dir}")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_hidden_states=False,
        class_weights=None,
        **kwargs,
    ):
        # if hasattr(self.bert, "gradient_checkpointing_disable") and self.bert.training:
        #     self.bert.gradient_checkpointing_disable()
        
        bert_kwargs = {}
        if token_type_ids is not None: bert_kwargs['token_type_ids'] = token_type_ids

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **bert_kwargs,
        )
        sequence_output = outputs[0]

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            if torch.any(lengths == 0):
                if get_rank_safe() <=0:
                    print("Предупреждение: Обнаружены последовательности с нулевой длиной в LSTM, attention_mask может быть некорректным.")
            
            packed_sequence = nn.utils.rnn.pack_padded_sequence(
                sequence_output, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_output_packed, _ = self.lstm(packed_sequence)
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_output_packed, batch_first=True, padding_value=0.0,
                total_length=input_ids.size(1)
            )
        else:
            lstm_output, _ = self.lstm(sequence_output)
        
        attention_scores = self.attention_layer(lstm_output) 
        if attention_mask is not None:
            if attention_scores.shape[1] != attention_mask.shape[1]:
                if get_rank_safe() <=0:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Несоответствие длин в LSTM attention! lstm_output: {attention_scores.shape[1]}, mask: {attention_mask.shape[1]}. Выравнивание может быть неточным.")
                squeezed_scores = attention_scores.squeeze(-1)
                squeezed_scores = squeezed_scores.masked_fill(attention_mask.eq(0), float("-inf"))
                attention_scores = squeezed_scores.unsqueeze(-1)

        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_output = lstm_output * attention_weights 
        combined_output = lstm_output + weighted_output
        transformed_output = self.transform_layer(combined_output)
        final_lstm_output = transformed_output

        final_lstm_output = self.dropout(final_lstm_output)
        logits = self.classifier(final_lstm_output)

        loss = None
        if labels is not None:
            crf_mask = (labels != -100).bool()
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = self.label2id.get('O', 0)
            try:
                loss = self.crf(logits, crf_labels, mask=crf_mask, reduction="mean")
            except ValueError as e:
                if get_rank_safe() <= 0:
                    print(f"Ошибка CRF (SpanBertLSTMForTokenClassification): {e}")
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not output_hidden_states:
            return {"loss": loss, "logits": logits}
        else:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": outputs.hidden_states,
            }

    def decode(self, logits, mask):
        return self.crf.decode(logits, mask=mask.bool() if mask is not None else None)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.bert, "gradient_checkpointing_enable"):
            self.bert.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self):
        if hasattr(self.bert, "gradient_checkpointing_disable"):
            self.bert.gradient_checkpointing_disable() 