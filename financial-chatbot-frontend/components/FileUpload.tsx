'use client';

import { useState } from 'react';

interface FileUploadProps {
  onDataLoaded: () => void;
}

export default function FileUpload({ onDataLoaded }: FileUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{
    type: 'success' | 'error' | null;
    message: string;
  }>({ type: null, message: '' });

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const fileType = file.name.split('.').pop()?.toLowerCase();
    if (!['csv', 'pdf', 'ppt', 'pptx'].includes(fileType || '')) {
      setUploadStatus({
        type: 'error',
        message: 'Invalid file type. Please upload CSV, PDF, or PPT files.',
      });
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', fileType || '');

    try {
      const response = await fetch('http://localhost:8000/api/upload/', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) throw new Error('Upload failed');
      
      const data = await response.json();
      setUploadStatus({
        type: 'success',
        message: `Successfully processed ${file.name}`,
      });
      onDataLoaded();
    } catch (error) {
      setUploadStatus({
        type: 'error',
        message: 'Error uploading file. Please try again.',
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-xl font-semibold mb-4">Upload Documents</h2>
      <div className="space-y-4">
        <label className="block">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center cursor-pointer hover:border-blue-500 transition-colors">
            <input
              type="file"
              className="hidden"
              accept=".csv,.pdf,.ppt,.pptx"
              onChange={handleFileUpload}
              disabled={uploading}
            />
            <span className="text-sm text-gray-600">
              {uploading ? 'Uploading...' : 'Click to upload or drag and drop'}
            </span>
          </div>
        </label>
        
        {uploadStatus.message && (
          <p className={`text-sm ${
            uploadStatus.type === 'success' ? 'text-green-600' : 'text-red-600'
          }`}>
            {uploadStatus.message}
          </p>
        )}
      </div>
    </div>
  );
} 