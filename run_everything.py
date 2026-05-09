#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 نظام الذكاء الاصطناعي المتكامل - كود واحد لكل شيء
AI Self-Improving System - All in One Code
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
from datetime import datetime
import json

# ============================================================================
# 1️⃣ إعداد النظام والـ Logger
# ============================================================================

class AILogger:
    """نظام التسجيل المتقدم"""
    def __init__(self, name="AI_System"):
        self.name = name
        self.logs = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def log(self, message, level="INFO"):
        log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {message}"
        print(log_entry)
        self.logs.append(log_entry)
    
    def info(self, message):
        self.log(f"ℹ️  {message}", "INFO")
    
    def success(self, message):
        self.log(f"✅ {message}", "SUCCESS")
    
    def warning(self, message):
        self.log(f"⚠️  {message}", "WARNING")
    
    def error(self, message):
        self.log(f"❌ {message}", "ERROR")

logger = AILogger("AI_System")

# ============================================================================
# 2️⃣ تعريف النماذج الثلاثة (V1, V2, V3)
# ============================================================================

class ModelV1(nn.Module):
    """النموذج الأساسي V1 - Simple Neural Network"""
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(ModelV1, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.version = "V1"
        logger.info(f"تم إنشاء النموذج {self.version}")
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ModelV2(nn.Module):
    """النموذج المحسّن V2 - مع Batch Normalization و Dropout"""
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(ModelV2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.version = "V2"
        logger.info(f"تم إنشاء النموذج {self.version}")
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

class ModelV3(nn.Module):
    """النموذج المتقدم V3 - مع Residual Connections"""
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(ModelV3, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.version = "V3"
        logger.info(f"تم إنشاء النموذج {self.version}")
    
    def forward(self, x):
        # الطبقة الأولى
        identity = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # الطبقات الوسطى مع Residual Connection
        residual = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = x + residual  # Residual Connection
        x = self.relu(x)
        x = self.dropout(x)
        
        # الطبقة الأخيرة
        x = self.fc4(x)
        return x

# ============================================================================
# 3️⃣ نظام التدريب المتقدم
# ============================================================================

class ModelTrainer:
    """فئة التدريب المتقدمة"""
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.history = {
            'losses': [],
            'accuracies': [],
            'epochs': 0
        }
    
    def train_epoch(self, train_loader, criterion, optimizer, epoch):
        """تدريب epoch واحد"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(self.device), y.to(self.device)
            
            # Forward pass
            outputs = self.model(X)
            loss = criterion(outputs, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # حساب الـ Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        self.history['losses'].append(avg_loss)
        self.history['accuracies'].append(accuracy)
        
        logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy
    
    def train(self, train_loader, epochs=10, lr=0.001):
        """تدريب كامل"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        logger.info(f"🚀 بدء التدريب - {epochs} epochs")
        logger.info(f"📊 حجم البيانات: {len(train_loader)} batches")
        
        for epoch in range(epochs):
            loss, acc = self.train_epoch(train_loader, criterion, optimizer, epoch)
        
        self.history['epochs'] = epochs
        logger.success(f"✨ انتهى التدريب! آخر Accuracy: {acc:.2f}%")
        return self.history

# ============================================================================
# 4️⃣ نظام التقييم والمقارنة الذاتية
# ============================================================================

class ModelEvaluator:
    """نظام التقييم الذاتي"""
    def __init__(self):
        self.results = {}
    
    def evaluate(self, model, X_test, y_test, model_name="Model"):
        """تقييم النموذج"""
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            loss = nn.CrossEntropyLoss()(outputs, y_test).item()
        
        self.results[model_name] = {
            'accuracy': accuracy,
            'loss': loss
        }
        
        logger.info(f"{model_name}: Accuracy={accuracy:.2f}%, Loss={loss:.4f}")
        return accuracy, loss
    
    def compare_models(self):
        """مقارنة جميع النماذج"""
        logger.info("⚖️  مقارنة النماذج:")
        print("\n" + "=" * 60)
        print("📊 نتائج المقارنة:")
        print("=" * 60)
        
        best_model = max(self.results, key=lambda x: self.results[x]['accuracy'])
        
        for model_name, metrics in self.results.items():
            accuracy = metrics['accuracy'] * 100
            loss = metrics['loss']
            status = "🏆 الأفضل!" if model_name == best_model else ""
            print(f"  {model_name}: {accuracy:.2f}% {status}")
        
        print("=" * 60 + "\n")
        return best_model

# ============================================================================
# 5️⃣ نظام التعلم المستمر
# ============================================================================

class ContinuousLearner:
    """نظام التعلم المستمر والتطوير الذاتي"""
    def __init__(self):
        self.knowledge_base = {
            'models': {},
            'best_model': None,
            'training_history': [],
            'improvements': []
        }
    
    def save_model_info(self, model_name, model, metrics):
        """حفظ معلومات النموذج"""
        self.knowledge_base['models'][model_name] = {
            'version': model.version,
            'accuracy': metrics[0],
            'loss': metrics[1],
            'timestamp': datetime.now().isoformat()
        }
        logger.success(f"📚 تم حفظ معلومات {model_name}")
    
    def generate_improvement(self, current_best, previous_best=None):
        """إنشاء تحسينات مستقبلية"""
        improvement = {
            'from': previous_best,
            'to': current_best,
            'timestamp': datetime.now().isoformat(),
            'description': f'تم التحسين من {previous_best} إلى {current_best}'
        }
        self.knowledge_base['improvements'].append(improvement)
        logger.success(f"🎯 {improvement['description']}")
    
    def display_knowledge_base(self):
        """عرض قاعدة المعرفة"""
        logger.info("📚 قاعدة المعرفة المتراكمة:")
        print("\n" + "=" * 60)
        print("🧠 معلومات النماذج:")
        print("=" * 60)
        
        for model_name, info in self.knowledge_base['models'].items():
            print(f"\n  📌 {model_name}:")
            print(f"     • Version: {info['version']}")
            print(f"     • Accuracy: {info['accuracy']:.2f}%")
            print(f"     • Loss: {info['loss']:.4f}")
            print(f"     • Saved at: {info['timestamp']}")
        
        print("\n" + "=" * 60)

# ============================================================================
# 6️⃣ البرنامج الرئيسي - تشغيل كل شيء
# ============================================================================

def main():
    """البرنامج الرئيسي الذي يشغل كل شيء"""
    
    print("\n" + "=" * 70)
    print("🤖 نظام الذكاء الاصطناعي المتكامل - AI Self-Improving System")
    print("=" * 70 + "\n")
    
    # الإعدادات
    DEVICE = 'cpu'  # أو 'cuda' إذا كان لديك GPU
    INPUT_DIM = 784
    HIDDEN_DIM = 128
    OUTPUT_DIM = 10
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 0.001
    
    logger.info(f"الجهاز المستخدم: {DEVICE}")
    logger.info(f"عدد الـ Epochs: {EPOCHS}")
    
    # ============================================================================
    # إنشاء البيانات التجريبية
    # ============================================================================
    
    logger.info("📊 جاري إنشاء البيانات التجريبية...")
    
    # بيانات التدريب
    X_train = torch.randn(500, INPUT_DIM)
    y_train = torch.randint(0, OUTPUT_DIM, (500,))
    
    # بيانات الاختبار
    X_test = torch.randn(100, INPUT_DIM)
    y_test = torch.randint(0, OUTPUT_DIM, (100,))
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    logger.success(f"✅ تم إنشاء البيانات")
    logger.info(f"   • Training: {X_train.shape[0]} samples")
    logger.info(f"   • Testing: {X_test.shape[0]} samples")
    
    # ============================================================================
    # إنشاء وتدريب النماذج
    # ============================================================================
    
    models_info = {
        'ModelV1': ModelV1(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM),
        'ModelV2': ModelV2(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM),
        'ModelV3': ModelV3(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM),
    }
    
    logger.info("🤖 جاري تدريب النماذج...")
    
    trainers = {}
    for model_name, model in models_info.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"تدريب {model_name}...")
        logger.info(f"{'='*60}")
        
        trainer = ModelTrainer(model, device=DEVICE)
        trainer.train(train_loader, epochs=EPOCHS, lr=LEARNING_RATE)
        trainers[model_name] = trainer
    
    # ============================================================================
    # تقييم النماذج
    # ============================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("🧪 جاري تقييم النماذج...")
    logger.info("=" * 70)
    
    evaluator = ModelEvaluator()
    
    for model_name, model in models_info.items():
        logger.info(f"\nتقييم {model_name}...")
        metrics = evaluator.evaluate(model, X_test, y_test, model_name)
    
    # ============================================================================
    # المقارنة والنتائج
    # ============================================================================
    
    best_model = evaluator.compare_models()
    
    # ============================================================================
    # نظام التعلم المستمر
    # ============================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("📚 حفظ المعرفة والتعلم المستمر...")
    logger.info("=" * 70)
    
    learner = ContinuousLearner()
    
    for model_name, metrics in evaluator.results.items():
        model = models_info[model_name]
        learner.save_model_info(model_name, model, metrics)
    
    learner.knowledge_base['best_model'] = best_model
    learner.generate_improvement(best_model)
    learner.display_knowledge_base()
    
    # ============================================================================
    # النتائج النهائية
    # ============================================================================
    
    print("\n" + "=" * 70)
    print("📈 النتائج النهائية:")
    print("=" * 70)
    
    print(f"\n🏆 أفضل نموذج: {best_model}")
    print(f"   • Accuracy: {evaluator.results[best_model]['accuracy']*100:.2f}%")
    print(f"   • Loss: {evaluator.results[best_model]['loss']:.4f}")
    
    print("\n📊 ملخص جميع النماذج:")
    for model_name, metrics in evaluator.results.items():
        print(f"   • {model_name}: {metrics['accuracy']*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("✨ تم الانتهاء من جميع العمليات بنجاح!")
    print("=" * 70 + "\n")

# ============================================================================
# تشغيل البرنامج
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("تم إيقاف البرنامج من قبل المستخدم")
    except Exception as e:
        logger.error(f"حدث خطأ: {str(e)}")
        import traceback
        traceback.print_exc()
