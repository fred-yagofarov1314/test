import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DistributedSampler
from contextlib import nullcontext
from typing import Optional, Callable, Dict, Union, Any

if not hasattr(__builtins__, "nullcontext"):

    class nullcontext(object):
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

try:
    from peft import (
        LoraConfig as PeftLoraConfig, 
        get_peft_model as peft_get_peft_model,
        TaskType as PeftTaskType,
        prepare_model_for_kbit_training as peft_prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
    print("--- Библиотека PEFT успешно импортирована. ---")
except ImportError:
    PEFT_AVAILABLE = False
    PeftLoraConfig = None
    peft_get_peft_model = None
    PeftTaskType = None
    peft_prepare_model_for_kbit_training = None
    print(
        "--- ВНИМАНИЕ: Библиотека PEFT не найдена. Установите ее: pip install peft ---"
    )

try:
    import bitsandbytes as bnb_lib 
    BNB_AVAILABLE = True
    print("--- Библиотека bitsandbytes успешно импортирована. ---")
except ImportError:
    BNB_AVAILABLE = False
    bnb_lib = None
    print(
        "--- ВНИМАНИЕ: bitsandbytes не найден. 4-битная квантизация и 8-битный оптимизатор могут быть недоступны. Установите: pip install bitsandbytes ---"
    )
# --- Конец проверок доступности библиотек ---


def get_rank_safe():
    """Безопасно получает ранг процесса для DDP."""
    return int(os.environ.get("RANK", -1))

def create_tensors_on_gpu(data, device="cuda"):
    """
    Создает тензоры сразу на GPU вместо их создания на CPU и последующего копирования.

    Аргументы:
        data: данные для преобразования в тензор
        device: устройство для создания тензора (по умолчанию 'cuda')

    Возвращает:
        PyTorch тензор на указанном устройстве
    """
    if isinstance(data, list) or isinstance(data, tuple):
        return [create_tensors_on_gpu(item, device) for item in data]
    elif isinstance(data, dict):
        return {
            key: create_tensors_on_gpu(value, device) for key, value in data.items()
        }
    elif isinstance(data, np.ndarray):
        return torch.as_tensor(data, device=device)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=True)
    else:
        return data


def optimize_cuda_for_t4():
    """Оптимизация параметров CUDA для максимальной производительности на T4 GPU"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
        print("--- CUDA оптимизировано для T4 GPU ---")
        return True
    return False


def optimize_cuda_settings():
    """Расширенные оптимизации CUDA для максимальной производительности при обучении"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.cuda, "memory_stats"):
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "max_memory_reserved"):
                try:
                    device = torch.device("cuda")
                    # total_memory = torch.cuda.get_device_properties(device).total_memory # Не используется
                    torch.cuda.set_per_process_memory_fraction(0.85)
                    print(
                        f"--- Зарезервировано 85% GPU памяти для кэша CUDA аллокатора ---"
                    )
                except Exception: # Более общее исключение
                    print(
                        "--- Не удалось зарезервировать память для CUDA аллокатора ---"
                    )
        if hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                print("--- Доступна оптимизация CUDA AMP ---")
            if hasattr(torch.cuda, "graph") and callable(
                getattr(torch.cuda, "graph", None)
            ):
                print("--- Доступна оптимизация CUDA графов ---")
        torch.autograd.profiler.emit_nvtx(False)
        torch.autograd.profiler.profile(False)
        print("--- Расширенные оптимизации CUDA активированы ---")
        return True
    return False


def create_optimized_dataloader(
    dataset, batch_size, is_training=True, world_size=1, rank=0, num_workers=0, collate_fn: Optional[Callable] = None
):
    """
    Создает оптимизированный DataLoader для обучения на T4 GPU
    """
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=is_training
        )
        shuffle = False
    else:
        shuffle = is_training and not isinstance(
            dataset, torch.utils.data.IterableDataset
        )
    
    prefetch_factor = 2 if num_workers > 0 else None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_training,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0,
        worker_init_fn=lambda worker_id: np.random.seed(
            np.random.get_state()[1][0] + worker_id
        ) if num_workers > 0 else None,
        collate_fn=collate_fn
    )


class DataPrefetcher:
    """
    Оптимизационный класс для асинхронной предзагрузки данных.
    """
    def __init__(self, dataloader, device="cuda", streams=None):
        self.device = device
        self.dataloader = dataloader
        self.iter = iter(dataloader)
        self.stream = torch.cuda.Stream() if device == "cuda" and torch.cuda.is_available() else None
        self.streams = streams if streams else {} 
        self.next_batch = None
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return

        if self.device != "cuda" or not torch.cuda.is_available() or self.stream is None:
            return

        with torch.cuda.stream(self.stream):
            if isinstance(self.next_batch, dict):
                for key, value in self.next_batch.items():
                    if isinstance(value, torch.Tensor):
                        self.next_batch[key] = value.to(self.device, non_blocking=True)
            elif isinstance(self.next_batch, (list, tuple)): 
                self.next_batch = [
                    (
                        t.to(self.device, non_blocking=True)
                        if isinstance(t, torch.Tensor)
                        else t
                    )
                    for t in self.next_batch
                ]
            
    def __iter__(self):
        return self

    def __next__(self):
        if self.device == "cuda" and torch.cuda.is_available() and self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = self.next_batch
            if batch is None:
                raise StopIteration
            
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                         value.record_stream(torch.cuda.current_stream())
            elif isinstance(batch, (list, tuple)):
                for t in batch:
                    if isinstance(t, torch.Tensor) and t.is_cuda:
                        t.record_stream(torch.cuda.current_stream())
            elif isinstance(batch, torch.Tensor) and batch.is_cuda:
                batch.record_stream(torch.cuda.current_stream())
        else:
            batch = self.next_batch
            if batch is None:
                raise StopIteration
        
        self.preload()
        return batch

    def __len__(self):
        return len(self.dataloader)


def enable_flash_attention():
    """
    Включает Flash Attention для ускорения внимания в Transformer моделях.
    """
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        try: 
            torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=True
            )
            print("--- Flash Attention успешно активирован ---")
            return True
        except Exception as e:
            print(f"--- Не удалось активировать Flash Attention через sdp_kernel: {e} ---")

    return False 


def enable_checkpointing_optimization(model, segment_size=2):
    """
    Включает оптимизированный gradient checkpointing.
    """
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False} 
            )
            print(f"--- Gradient checkpointing включен для transformers модели ---")
            return True
        elif hasattr(model, "config") and hasattr(model.config, "gradient_checkpointing"):
            model.config.gradient_checkpointing = True
            if hasattr(model, "bert") and hasattr(model.bert, "gradient_checkpointing_enable"):
                 model.bert.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            print(f"--- Gradient checkpointing включен через model.config ---")
            return True

        pass

        print(
            "--- Не удалось включить gradient checkpointing: модель не поддерживает стандартные методы. ---"
        )
        return False
    except Exception as e:
        print(f"--- Ошибка при включении gradient checkpointing: {e} ---")
        return False

# --- Утилитарная функция для инициализации матрицы переходов CRF (для CustomCRF или torchcrf) ---
def initialize_crf_transitions_bio(crf_layer, id2label, label2id, is_custom_crf=False):
    """
    Инициализирует матрицу переходов CRF с учетом правил BIO-формата.
    Работает как с torchcrf.CRF, так и с CustomCRF.
    """
    if not hasattr(crf_layer, 'transitions'):
        print("--- Слой CRF не имеет атрибута 'transitions'. Инициализация пропущена. ---")
        return

    transitions = crf_layer.transitions.data
    num_labels = transitions.shape[0]

    ALLOWED_TRANSITION_SCORE = 5.0
    FORBIDDEN_TRANSITION_SCORE = -5.0

    if is_custom_crf and hasattr(crf_layer, 'start_transitions') and hasattr(crf_layer, 'end_transitions'):
        for i in range(num_labels):
            label = id2label.get(i)
            if label:
                if label.startswith('B-') or label == 'O':
                    crf_layer.start_transitions.data[i] = ALLOWED_TRANSITION_SCORE / 2.0 
                elif label.startswith('I-'):
                    crf_layer.start_transitions.data[i] = FORBIDDEN_TRANSITION_SCORE
        
    for i in range(num_labels): 
        for j in range(num_labels): 
            label_from = id2label.get(i)
            label_to = id2label.get(j)

            if not label_from or not label_to:
                continue

            if label_from == 'O':
                if label_to.startswith('I-'): 
                    transitions[i, j] = FORBIDDEN_TRANSITION_SCORE
                elif label_to.startswith('B-') or label_to == 'O':
                     pass 

            elif label_from.startswith('B-'):
                entity_type_from = label_from[2:]
                if label_to.startswith('I-'):
                    entity_type_to = label_to[2:]
                    if entity_type_from == entity_type_to: 
                        transitions[i, j] = ALLOWED_TRANSITION_SCORE
                    else: 
                        transitions[i, j] = FORBIDDEN_TRANSITION_SCORE
                elif label_to.startswith('B-'): 
                    pass
                elif label_to == 'O': 
                    pass
            
            elif label_from.startswith('I-'):
                entity_type_from = label_from[2:]
                if label_to.startswith('I-'):
                    entity_type_to = label_to[2:]
                    if entity_type_from == entity_type_to: 
                        transitions[i, j] = ALLOWED_TRANSITION_SCORE
                    else: 
                        transitions[i, j] = FORBIDDEN_TRANSITION_SCORE
                elif label_to.startswith('B-'): 
                     pass
                elif label_to == 'O': 
                    pass
    
    crf_layer.transitions.data = transitions
    if get_rank_safe() <=0:
        print(f"--- Матрица переходов CRF ({'CustomCRF' if is_custom_crf else 'TorchCRF'}) инициализирована с правилами BIO. ---")


ALL_ENTITY_TYPES = [
    "SURNAME", "NAME", "PATRONYMIC", "INDIVIDUAL_TAX_ID", "SNILS", "USERNAME", 
    "PASSWORD", "PIN", "TOKEN", "EMAIL", "MOBILE_PHONE", "LANDLINE_PHONE", 
    "INTERNATIONAL_PHONE", "COUNTRY", "REGION", "CITY", "DISTRICT", "LOCALITY", 
    "STREET", "COORDINATES", "POSTAL_CODE", "ORGANIZATION", "DEPARTMENT", 
    "AUTHORITY", "SUBDIVISION", "OGRN", "OGRNIP", "LEGAL_TAX_ID", "DEVICE_ID", 
    "SERIAL_NUMBER", "IMEI", "MAC", "IBAN", "CARD_NUMBER", "BIK", "ACCOUNT", 
    "LICENSE_PLATE", "KPP", "OMS_POLICY_NUMBER", "CARD_TYPE", "NUMBER", "IPV4", 
    "IPV6", "URL", "DATE_TIME", "DATE_MONTH", "DATE_YEAR", "DATE_DAY",
]

def get_all_bio_labels(entity_types):
    """Генерирует список всех BIO-меток на основе списка типов сущностей."""
    bio_labels = ["O"] # O is first
    for entity_type in entity_types:
        bio_labels.append(f"B-{entity_type}")
        bio_labels.append(f"I-{entity_type}")
    
    # Чтобы 'O' гарантированно имел ID 0, а остальные были отсортированы:
    unique_tags = list(set(bio_labels))
    if "O" in unique_tags:
        unique_tags.remove("O")
        return ["O"] + sorted(unique_tags)
    else:
        # Этого не должно произойти, но на всякий случай
        return sorted(unique_tags)


def read_conll(path):
    """Читает данные в формате CoNLL."""
    tokens_list, labels_list = [], []
    current_tokens, current_labels = [], []
    line_num = 0
    
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line: 
                    if current_tokens: 
                        tokens_list.append(current_tokens)
                        labels_list.append(current_labels)
                        current_tokens, current_labels = [], []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        current_tokens.append(parts[0])
                        current_labels.append(parts[1])
                    else:
                        if get_rank_safe() <= 0: 
                            pass
        
        if current_tokens:
            tokens_list.append(current_tokens)
            labels_list.append(current_labels)
    except FileNotFoundError:
        if get_rank_safe() <= 0:
            print(f"ОШИБКА: Файл датасета не найден по пути: {path}")
        return [], [] # Возвращаем пустые списки без valid_labels
    except Exception as e:
        if get_rank_safe() <= 0:
            print(f"ОШИБКА при чтении файла {path}: {e}")
        return [], [] # Возвращаем пустые списки без valid_labels

    # --- ДОБАВЛЕНО ДЛЯ ОТЛАДКИ ---
    if get_rank_safe() <= 0 and len(labels_list) > 5:
        pass
    # --- КОНЕЦ ОТЛАДКИ ---

    # Удалена неиспользуемая логика проверки BIO-меток и сбора valid_labels
    return tokens_list, labels_list 