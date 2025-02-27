// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add header to the page
    addHeader();
    
    // Initialize fade-in animations
    initFadeAnimations();
    
    // Add hover animations to buttons
    addButtonAnimations();
    
    // Add scroll animations
    initScrollAnimations();
    
    // Enhance loading spinner
    enhanceLoadingSpinner();
    
    // Add tooltips to buttons
    addTooltips();
    
    // Add smooth scrolling
    addSmoothScrolling();
    
    // Add DataTable enhancements
    enhanceDataTables();
});

// Add a header to the page
function addHeader() {
    // Create header element
    const header = document.createElement('div');
    header.className = 'header-container';
    
    // Add header content
    header.innerHTML = `
        <h1 class="header-title">Drug Discovery Platform</h1>
        <nav class="header-nav">
            <a href="#">Home</a>
            <a href="#">Documentation</a>
            <a href="#">About</a>
            <a href="#">Contact</a>
        </nav>
    `;
    
    // Insert at the beginning of the body
    document.body.insertBefore(header, document.body.firstChild);
}

// Initialize fade-in animations for elements as they come into view
function initFadeAnimations() {
    // Create an observer for elements that should fade in
    const fadeObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                fadeObserver.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    // Observe target info and optimization results
    const fadeElements = [
        document.getElementById('targetInfo'),
        document.getElementById('optimizationResults'),
        document.getElementById('viewer'),
        document.getElementById('viewer3D')
    ];
    
    fadeElements.forEach(el => {
        if (el) fadeObserver.observe(el);
    });
    
    // Add fade-in class to the table when it's populated
    const originalLoadCompounds = window.loadCompounds;
    if (originalLoadCompounds) {
        window.loadCompounds = function() {
            originalLoadCompounds.apply(this, arguments);
            const table = document.getElementById('compoundsTable');
            if (table) {
                setTimeout(() => {
                    table.classList.add('fade-in');
                }, 300);
            }
        };
    }
}

// Add hover animations to buttons
function addButtonAnimations() {
    const buttons = document.querySelectorAll('button');
    
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.1)';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
    });
}

// Initialize scroll animations
function initScrollAnimations() {
    let lastScrollTop = 0;
    
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const header = document.querySelector('.header-container');
        
        // Header shrink effect on scroll
        if (header) {
            if (scrollTop > 50) {
                header.style.padding = '0.5rem 2rem';
                header.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
            } else {
                header.style.padding = '1rem 2rem';
                header.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
            }
        }
        
        // Reveal elements on scroll
        const revealElements = document.querySelectorAll('#targetInfo, #optimizationResults, #viewer, #viewer3D');
        revealElements.forEach(el => {
            if (!el) return;
            
            const elementTop = el.getBoundingClientRect().top;
            const elementVisible = 150;
            
            if (elementTop < window.innerHeight - elementVisible) {
                el.classList.add('fade-in');
            }
        });
        
        lastScrollTop = scrollTop;
    });
}

// Enhance the loading spinner
function enhanceLoadingSpinner() {
    const originalShowLoading = window.showLoading;
    if (originalShowLoading) {
        window.showLoading = function(show) {
            originalShowLoading.apply(this, arguments);
            const loadingElement = document.getElementById('loading');
            if (loadingElement) {
                if (show) {
                    loadingElement.classList.add('visible');
                } else {
                    loadingElement.classList.remove('visible');
                }
            }
        };
    }
}

// Add tooltips to buttons
function addTooltips() {
    const buttons = {
        'Search Target': 'Search for a protein target by ID or name',
        'Load Compounds': 'Load compounds for the selected target',
        'View 3D': 'View the 3D structure of the molecule',
        'Optimize': 'Optimize the selected lead compound'
    };
    
    document.querySelectorAll('button').forEach(button => {
        const buttonText = button.textContent.trim();
        if (buttons[buttonText]) {
            button.setAttribute('title', buttons[buttonText]);
            
            // Add tooltip functionality
            button.addEventListener('mouseenter', function(e) {
                const tooltip = document.createElement('div');
                tooltip.className = 'tooltip';
                tooltip.textContent = this.getAttribute('title');
                tooltip.style.position = 'absolute';
                tooltip.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
                tooltip.style.color = 'white';
                tooltip.style.padding = '5px 10px';
                tooltip.style.borderRadius = '4px';
                tooltip.style.fontSize = '0.8rem';
                tooltip.style.zIndex = '1000';
                tooltip.style.top = (e.target.getBoundingClientRect().bottom + 5) + 'px';
                tooltip.style.left = (e.target.getBoundingClientRect().left) + 'px';
                tooltip.style.opacity = '0';
                tooltip.style.transition = 'opacity 0.3s ease';
                
                document.body.appendChild(tooltip);
                
                setTimeout(() => {
                    tooltip.style.opacity = '1';
                }, 10);
                
                this.addEventListener('mouseleave', function() {
                    tooltip.style.opacity = '0';
                    setTimeout(() => {
                        document.body.removeChild(tooltip);
                    }, 300);
                }, { once: true });
            });
        }
    });
}

// Add smooth scrolling
function addSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 70,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Enhance DataTables
function enhanceDataTables() {
    // Check if DataTable is initialized
    const checkDataTable = setInterval(() => {
        const dataTable = document.querySelector('.dataTable');
        if (dataTable) {
            clearInterval(checkDataTable);
            
            // Add responsive classes
            const tableWrapper = document.querySelector('.dataTables_wrapper');
            if (tableWrapper) {
                tableWrapper.classList.add('table-responsive');
            }
            
            // Style pagination
            const pagination = document.querySelector('.dataTables_paginate');
            if (pagination) {
                pagination.classList.add('pagination-custom');
            }
        }
    }, 1000);
}

// Page transition effect
function pageTransition() {
    const overlay = document.createElement('div');
    overlay.classList.add('page-transition-overlay');
    document.body.appendChild(overlay);
    
    setTimeout(() => {
        overlay.style.opacity = '1';
        
        setTimeout(() => {
            overlay.style.opacity = '0';
            setTimeout(() => {
                document.body.removeChild(overlay);
            }, 500);
        }, 500);
    }, 10);
}

// Add page transition to links
document.addEventListener('click', function(e) {
    const target = e.target.closest('a');
    if (target && !target.getAttribute('href').startsWith('#') && target.getAttribute('target') !== '_blank') {
        e.preventDefault();
        pageTransition();
        setTimeout(() => {
            window.location.href = target.getAttribute('href');
        }, 500);
    }
});