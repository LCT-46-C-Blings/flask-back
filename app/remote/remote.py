import requests
import json
from typing import List, Dict, Optional, Any
from datetime import datetime, date

class RemoteClient:
    def __init__(self, base_url: str = "https://interpersonal-jody-unsanctionable.ngrok-free.dev"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "MedicalClient/1.0"
        })
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Обработка ответа от API"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            raise
    
    # Patient operations
    def get_all_patients(self) -> List[Dict]:
        """Получить всех пациентов"""
        response = self.session.get(f"{self.base_url}/patients")
        return self._handle_response(response)
    
    def get_patient(self, patient_id: int) -> Dict:
        """Получить пациента по ID"""
        response = self.session.get(f"{self.base_url}/patients/{patient_id}")
        return self._handle_response(response)
    
    def create_patient(self, patient_data: Dict) -> Dict:
        """Создать нового пациента"""
        response = self.session.post(
            f"{self.base_url}/patients", 
            json=patient_data
        )
        return self._handle_response(response)
    
    def update_patient(self, patient_id: int, patient_data: Dict) -> Dict:
        """Обновить данные пациента"""
        response = self.session.put(
            f"{self.base_url}/patients/{patient_id}", 
            json=patient_data
        )
        return self._handle_response(response)
    
    def delete_patient(self, patient_id: int) -> bool:
        """Удалить пациента"""
        response = self.session.delete(f"{self.base_url}/patients/{patient_id}")
        try:
            response.raise_for_status()
            return True
        except requests.exceptions.HTTPError:
            return False
    
    def get_patient_appointments(self, patient_id: int) -> List[Dict]:
        """Получить приемы пациента"""
        response = self.session.get(f"{self.base_url}/patients/{patient_id}/appointments")
        return self._handle_response(response)
    
    def get_patient_medical_history(self, patient_id: int) -> List[Dict]:
        """Получить анамнез пациента"""
        response = self.session.get(f"{self.base_url}/patients/{patient_id}/medical_history")
        return self._handle_response(response)
    
    # Doctor operations
    def get_all_doctors(self) -> List[Dict]:
        """Получить всех врачей"""
        response = self.session.get(f"{self.base_url}/doctors")
        return self._handle_response(response)
    
    def get_doctor(self, doctor_id: int) -> Dict:
        """Получить врача по ID"""
        response = self.session.get(f"{self.base_url}/doctors/{doctor_id}")
        return self._handle_response(response)
    
    def get_doctor_appointments(self, doctor_id: int) -> List[Dict]:
        """Получить приемы врача"""
        response = self.session.get(f"{self.base_url}/doctors/{doctor_id}/appointments")
        return self._handle_response(response)
    
    # Appointment operations
    def get_all_appointments(self, patient_id: Optional[int] = None, doctor_id: Optional[int] = None) -> List[Dict]:
        """Получить все приемы (с фильтрацией)"""
        params = {}
        if patient_id:
            params["patient_id"] = patient_id
        if doctor_id:
            params["doctor_id"] = doctor_id
            
        response = self.session.get(f"{self.base_url}/appointments", params=params)
        return self._handle_response(response)
    
    def get_appointment(self, appointment_id: int) -> Dict:
        """Получить прием по ID"""
        response = self.session.get(f"{self.base_url}/appointments/{appointment_id}")
        return self._handle_response(response)
    
    def create_appointment(self, appointment_data: Dict) -> Dict:
        """Создать новый прием"""
        response = self.session.post(
            f"{self.base_url}/appointments", 
            json=appointment_data
        )
        return self._handle_response(response)
    
    def update_appointment(self, appointment_id: int, appointment_data: Dict) -> Dict:
        """Обновить прием"""
        response = self.session.put(
            f"{self.base_url}/appointments/{appointment_id}", 
            json=appointment_data
        )
        return self._handle_response(response)
    
    def delete_appointment(self, appointment_id: int) -> bool:
        """Удалить прием"""
        response = self.session.delete(f"{self.base_url}/appointments/{appointment_id}")
        try:
            response.raise_for_status()
            return True
        except requests.exceptions.HTTPError:
            return False
    
    # Medical History operations
    def get_all_medical_history(self, patient_id: Optional[int] = None, history_type: Optional[str] = None) -> List[Dict]:
        """Получить анамнез (с фильтрацией)"""
        params = {}
        if patient_id:
            params["patient_id"] = patient_id
        if history_type:
            params["type"] = history_type
            
        response = self.session.get(f"{self.base_url}/medical_history", params=params)
        return self._handle_response(response)
    
    def create_medical_history(self, history_data: Dict) -> Dict:
        """Создать запись анамнеза"""
        response = self.session.post(
            f"{self.base_url}/medical_history", 
            json=history_data
        )
        return self._handle_response(response)
    
    def delete_medical_history(self, history_id: int) -> bool:
        """Удалить запись анамнеза"""
        response = self.session.delete(f"{self.base_url}/medical_history/{history_id}")
        try:
            response.raise_for_status()
            return True
        except requests.exceptions.HTTPError:
            return False