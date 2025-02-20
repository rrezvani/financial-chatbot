'use client';

import React, { useState, useEffect, useRef } from 'react';

interface TaxDetails {
  income_source: string;
  deduction_type: string;
  deductions: string;
  taxable_income: string;
  tax_owed: string;
  transaction_date: string;
}

interface EnhancedResult {
  type: string;
  income_range: string;
  rate: string;
  conditions: string;
  details?: TaxDetails;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  enhancedResults?: EnhancedResult[];
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    console.log('Submitting message:', input);

    const userMessage: Message = { role: 'user' as const, content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/chat/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: input }),
      });

      const data = await response.json();
      console.log('Received response:', data);

      if (data.message) {
        const botMessage: Message = {
          role: 'assistant' as const,
          content: data.message,
          enhancedResults: data.enhanced_results
        };
        setMessages(prev => [...prev, botMessage]);
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant' as const,
        content: 'Sorry, I encountered an error. Please try again.'
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div key={index} 
               className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-4 rounded-lg ${
              message.role === 'user' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-700 text-white'
            }`}>
              <div className="whitespace-pre-wrap font-sans">{message.content}</div>
              
              {message.enhancedResults && message.enhancedResults.length > 0 && (
                <div className="mt-4 space-y-3">
                  {message.enhancedResults.map((result, idx) => (
                    <div key={idx} className="border-t border-gray-600 pt-3">
                      <div className="font-semibold text-blue-300">
                        Income Range: {result.income_range}
                      </div>
                      <div className="text-green-300">
                        Tax Rate: {result.rate}
                      </div>
                      <div className="text-sm text-gray-300">
                        {result.conditions}
                      </div>
                      
                      {result.details && (
                        <div className="mt-2 text-sm space-y-1 bg-gray-800 p-2 rounded">
                          <div className="text-blue-200">
                            Income Source: {result.details.income_source}
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            <div>Deductions: {result.details.deductions}</div>
                            <div>Type: {result.details.deduction_type}</div>
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            <div>Taxable Income: {result.details.taxable_income}</div>
                            <div>Tax Owed: {result.details.tax_owed}</div>
                          </div>
                          <div className="text-gray-400">
                            Transaction Date: {result.details.transaction_date}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-700 text-white p-4 rounded-lg animate-pulse">
              Thinking...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-700">
        <div className="flex space-x-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about taxes..."
            className="flex-1 p-3 rounded bg-gray-800 text-white border border-gray-700 focus:outline-none focus:border-blue-500"
          />
          <button 
            type="submit"
            disabled={loading}
            className="px-6 py-3 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </form>
    </div>
  );
} 